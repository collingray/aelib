from dataclasses import dataclass
from typing import Optional, List, Literal

import torch.optim.lr_scheduler
import wandb

from autoencoder_multilayer import *
from buffer import ActivationsBuffer
from utils import *


@dataclass
class AutoEncoderMultiLayerTrainerConfig:
    """
    The configuration for the `AutoEncoderMultiLayerTrainer` class

    lr: the learning rate to use
    beta1: beta1 for adam
    beta2: beta2 for adam
    total_steps: the total number of steps that will be taken by the lr scheduler, should be equal to the number of times train_on is called
    warmup_percent: the percentage of steps to use for warmup
    wb_project: the wandb project to log to
    wb_entity: the wandb entity to log to
    wb_group: the wandb group to log to
    wb_config: the wandb config to log
    """
    lr: float
    beta1: float
    beta2: float
    total_steps: int
    warmup_percent: float
    steps_per_report: int = 128
    steps_per_resample: Optional[int] = None
    num_resamples: Optional[int] = None
    wb_project: Optional[str] = None
    wb_entity: Optional[str] = None
    wb_name: Optional[str] = None
    wb_group: Optional[str] = None
    wb_config: Optional[dict] = None


class AutoEncoderMultiLayerTrainer:
    """
    The class for training an `AutoEncoderMultiLayer` model, which contains the model and optimizer, and trains on
    activations passed to it through the `train_on` method, but does not contain the data or training loop
    """

    def __init__(self, encoder_cfg: AutoEncoderMultiLayerConfig, trainer_cfg: AutoEncoderMultiLayerTrainerConfig):
        self.cfg = trainer_cfg
        self.steps = 0

        self.encoder = AutoEncoderMultiLayer(encoder_cfg)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=trainer_cfg.lr,
            betas=(trainer_cfg.beta1, trainer_cfg.beta2),
            foreach=False
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=trainer_cfg.lr,
            total_steps=trainer_cfg.total_steps,
            pct_start=trainer_cfg.warmup_percent
        )

        wandb.init(
            project=trainer_cfg.wb_project,
            entity=trainer_cfg.wb_entity,
            name=trainer_cfg.wb_name,
            group=trainer_cfg.wb_group,
            config=trainer_cfg.wb_config,
            settings=wandb.Settings(disable_job_creation=True)
        )

    def train_on(self, acts, buffer: Optional[ActivationsBuffer] = None):  # acts: [batch_size, num_layers, n_dim]
        enc, loss, l1, mse = self.encoder(acts)  # loss: [num_layers]
        loss.mean().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.steps += 1

        if (self.cfg.steps_per_resample and self.steps % self.cfg.steps_per_resample == 0) and (
                not self.cfg.num_resamples or self.steps // self.cfg.steps_per_resample <= self.cfg.num_resamples):
            assert buffer is not None, "Buffer must be provided to resample neurons"

            inputs = buffer.next(batch=2 ** 14).to(device=self.encoder.cfg.device)
            self.encoder.resample_neurons(inputs, batch_size=2 ** 12, optimizer=self.optimizer)

        if self.steps % self.cfg.steps_per_report == 0:
            metrics = {}

            for layer in range(self.encoder.num_layers):
                metrics[f"layer_{layer}"] = {
                    "l1": l1[layer].item(),
                    "mse": mse[layer].item(),
                    "loss": loss[layer].item(),
                }

            if self.encoder.cfg.record_data:
                freqs, avg_l0, avg_fvu = self.encoder.get_data()
                for layer in range(self.encoder.num_layers):
                    metrics[f"layer_{layer}"].update({
                        "feature_density": wandb.Histogram(freqs[layer].log10().nan_to_num(neginf=-10).cpu()),
                        "avg_l0": avg_l0[layer],
                        "avg_fvu": avg_fvu[layer]
                    })

                metrics.update({
                    "feature_density": wandb.Histogram(freqs.mean(dim=0).log10().nan_to_num(neginf=-10).cpu()),
                    "avg_l0": avg_l0.mean(),
                    "avg_fvu": avg_fvu.mean()
                })

            wandb.log({
                "lr": self.scheduler.get_last_lr()[0],
                **metrics,
            })

    def finish(self):
        # Log the final data if it was recorded, then finish the wandb run
        if self.encoder.cfg.record_data:
            freqs, avg_l0, avg_fvu = self.encoder.get_data()
            wandb.log({
                "feature_density": wandb.Histogram(freqs.mean(dim=0).log10().nan_to_num(neginf=-10).cpu()),
                "avg_l0": avg_l0.mean(),
                "avg_fvu": avg_fvu.mean()
            })

        wandb.finish()
