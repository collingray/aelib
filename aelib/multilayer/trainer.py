from dataclasses import dataclass
from typing import Optional, List, Literal

import torch.optim.lr_scheduler
import wandb

from .autoencoder import AutoEncoderMultiLayer, AutoEncoderMultiLayerConfig
from ..buffer import ActivationsBuffer
from ..utils import plateau_lr_scheduler


@dataclass
class AutoEncoderMultiLayerTrainerConfig:
    """
    The configuration for the `AutoEncoderMultiLayerTrainer` class

    lr: the learning rate to use
    beta1: beta1 for adam
    beta2: beta2 for adam
    l1_weight: the l1 loss weight
    total_steps: the total number of steps that will be taken by the lr scheduler, should be equal to the number of times train_on is called
    lr_warmup_pct: the percentage of steps to use for warmup
    lr_decay_pct: the percentage of steps to use for decay
    l1_warmup_pct: the percentage of steps to use for l1 warmup
    steps_per_report: the number of steps between each report
    steps_per_resample: the number of steps between each resample, or None to disable resampling
    num_resamples: the number of resamples to perform
    decoder_norm_scale: whether to scale the sparsity loss by the l2 norm of the decoder weights
    wb_project: the wandb project to log to
    wb_entity: the wandb entity to log to
    wb_name: the wandb run name
    wb_group: the wandb group to log to
    wb_config: the wandb config to log
    """
    lr: float
    beta1: float
    beta2: float
    l1_weight: float
    total_steps: int
    lr_warmup_pct: float
    lr_decay_pct: float
    l1_warmup_pct: Optional[float]
    steps_per_report: int = 128
    steps_per_resample: Optional[int] = None
    num_resamples: Optional[int] = None
    decoder_norm_scale: bool = False
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
        self.step = 0

        self.l1_weight = trainer_cfg.l1_weight if trainer_cfg.l1_warmup_pct is None else 0.0
        self.l1_inc = 0.0
        if trainer_cfg.l1_warmup_pct is not None:
            self.l1_inc = trainer_cfg.l1_weight / (trainer_cfg.total_steps * trainer_cfg.l1_warmup_pct)

        self.encoder = AutoEncoderMultiLayer(encoder_cfg)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=trainer_cfg.lr,
            betas=(trainer_cfg.beta1, trainer_cfg.beta2),
            foreach=False
        )

        self.scheduler = plateau_lr_scheduler(
            self.optimizer,
            total_steps=trainer_cfg.total_steps,
            warmup_pct=trainer_cfg.lr_warmup_pct,
            decay_pct=trainer_cfg.lr_decay_pct,
        )

        if trainer_cfg.wb_project is not None and trainer_cfg.wb_entity is not None:
            wandb.init(
                project=trainer_cfg.wb_project,
                entity=trainer_cfg.wb_entity,
                name=trainer_cfg.wb_name,
                group=trainer_cfg.wb_group,
                config=trainer_cfg.wb_config,
                settings=wandb.Settings(disable_job_creation=True)
            )

    def train_on(self, acts, buffer: Optional[ActivationsBuffer] = None):  # acts: [batch_size, num_layers, n_dim]
        enc, loss, l1, mse = self.encoder(acts, self.l1_weight, decoder_norm_scale=self.cfg.decoder_norm_scale)
        loss.mean().backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.step += 1
        self.l1_weight = min(self.l1_weight + self.l1_inc, self.cfg.l1_weight)

        if (self.cfg.steps_per_resample and self.step % self.cfg.steps_per_resample == 0) and (
                not self.cfg.num_resamples or self.step // self.cfg.steps_per_resample <= self.cfg.num_resamples):
            assert buffer is not None, "Buffer must be provided to resample neurons"

            inputs = buffer.next(batch=2 ** 14).to(device=self.encoder.cfg.device)
            self.encoder.resample_neurons(inputs, batch_size=2 ** 12, optimizer=self.optimizer)

        if self.step % self.cfg.steps_per_report == 0:
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

                metrics["total"] = {
                    "feature_density": wandb.Histogram(freqs.mean(dim=0).log10().nan_to_num(neginf=-10).cpu())
                }

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