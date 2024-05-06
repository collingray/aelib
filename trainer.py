from dataclasses import dataclass
from typing import Optional

import torch.optim.lr_scheduler
import wandb

from autoencoder import AutoEncoder, AutoEncoderConfig
from utils import plateau_lr_scheduler


@dataclass
class AutoEncoderTrainerConfig:
    """
    The configuration for the `AutoEncoderTrainer` class

    lr: the learning rate to use
    beta1: beta1 for adam
    beta2: beta2 for adam
    total_steps: the total number of steps that will be taken by the lr scheduler, should be equal to the number of times train_on is called
    warmup_pct: the percentage of steps to use for warmup
    decay_pct: the percentage of steps to use for decay
    wb_project: the wandb project to log to
    wb_entity: the wandb entity to log to
    wb_group: the wandb group to log to
    wb_config: the wandb config to log
    """
    lr: float
    beta1: float
    beta2: float
    total_steps: int
    warmup_pct: float
    decay_pct: float
    wb_project: str
    wb_entity: str
    wb_name: Optional[str] = None
    wb_group: Optional[str] = None
    wb_config: Optional[dict] = None
    steps_per_report: int = 100


class AutoEncoderTrainer:
    """
    The class for training an `AutoEncoder` model, which contains the model and optimizer, and trains on
    activations passed to it through the `train_on` method, but does not contain the data or training loop
    """

    def __init__(self, encoder_cfg: AutoEncoderConfig, trainer_cfg: AutoEncoderTrainerConfig):
        self.cfg = trainer_cfg
        self.steps = 0

        self.encoder = AutoEncoder(encoder_cfg)

        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=trainer_cfg.lr,
            betas=(trainer_cfg.beta1, trainer_cfg.beta2),
            foreach=False
        )

        self.scheduler = plateau_lr_scheduler(
            self.optimizer,
            total_steps=trainer_cfg.total_steps,
            warmup_pct=trainer_cfg.warmup_pct,
            decay_pct=trainer_cfg.decay_pct,
        )

        wandb.init(
            project=trainer_cfg.wb_project,
            entity=trainer_cfg.wb_entity,
            name=trainer_cfg.wb_name,
            group=trainer_cfg.wb_group,
            config=trainer_cfg.wb_config,
            settings=wandb.Settings(disable_job_creation=True)
        )

    def train_on(self, acts):
        enc, loss, l1, mse = self.encoder(acts)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

        self.steps += 1

        if self.steps % self.cfg.steps_per_report == 0:
            if self.encoder.cfg.record_data:
                freqs, avg_l0, avg_fvu = self.encoder.get_data()
                freq_data = {
                    "feature_density": wandb.Histogram(freqs.log10().nan_to_num(neginf=-10).cpu()),
                    "avg_l0": avg_l0,
                    "avg_fvu": avg_fvu
                }
            else:
                freq_data = {}

            wandb.log({
                "l1": l1.item(),
                "mse": mse.item(),
                "loss": loss.item(),
                "lr": self.scheduler.get_last_lr()[0],
                **freq_data,
            })

    def finish(self):
        # Log the final data if it was recorded, then finish the wandb run
        if self.encoder.cfg.record_data:
            freqs, avg_l0, avg_fvu = self.encoder.get_data()
            wandb.log({
                "feature_density": wandb.Histogram(freqs.log10().nan_to_num(neginf=-10).cpu()),
                "avg_l0": avg_l0,
                "avg_fvu": avg_fvu
            })

        wandb.finish()
