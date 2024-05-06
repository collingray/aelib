import multiprocessing
from typing import List, Optional, Literal

from autoencoder_trainer import *
from autoencoder_multilayer_trainer import *
from buffer import *
from tqdm.autonotebook import tqdm
from utils import *
import gc
import torch.multiprocessing
from torch.multiprocessing import JoinableQueue as Queue, spawn
import logging
import sys
import traceback


@dataclass
class AutoEncoderSweeperConfig:
    """
    The configuration for the `AutoEncoderSweeper` class

    n_dim: the dimension of the input
    m_dim: the dimension of the hidden layer
    lr: the learning rate to use
    beta1: beta1 for adam
    beta2: beta2 for adam
    lambda_reg: the regularization strength to use for the autoencoder
    warmup_percent: the percentage of steps to use for warmup
    dtype: the dtype to use for the model and autoencoder
    device: the device to use for the model and autoencoder
    layer: the layer of the model to train the autoencoder on (0-indexed)
    total_activations: the total number of activations to train each autoencoder on
    batch_size: the batch size to use for training each autoencoder
    parallelism: the number of autoencoders to train in parallel
    steps_per_report: the number of steps to take before reporting to wandb
    wb_project: the wandb project to log to
    wb_entity: the wandb entity to log to
    wb_group: the wandb group to log to
    wb_config: the wandb config to log
    act_norms: the norms to use for layer activation renormalization. If set, then AutoEncoderMultiLayer will be used,
        otherwise AutoEncoder will be used
    act_renorm_type: the type of renormalization to use for layer activations, one of "linear", "sqrt",
        "log", "none". Activations are scaled by act_renorm_scale*(avg(norms)/norms[layer]), where norms are the
        result of the act_renorm_type applied to act_norms. Only used if act_norms is set
    act_renorm_scale: a global scale to apply to all activations after renormalization. Only used if act_norms is set
    """
    n_dim: int
    m_dim: int
    lr: List[float]
    beta1: List[float]
    beta2: List[float]
    lambda_reg: List[float]
    warmup_percent: List[float]
    layer: List[int]
    act_norms: Optional[List[float]]
    act_renorm_type: List[Literal["linear", "sqrt", "log", "none"]]
    act_renorm_scale: List[float]
    wb_project: str
    wb_entity: str
    wb_group: Optional[str] = None
    wb_config: Optional[dict] = None
    dtype: torch.dtype = torch.bfloat16
    device: str = "cuda"
    total_activations: int = int(2e7)
    batch_size: int = 1024
    parallelism: int = 8
    steps_per_report: int = 100


def create_trainer_worker(pidx: int, offset: int, sweep_cfgs: list[dict], act_queues: list[Queue],
                          cfg: AutoEncoderSweeperConfig):
    sweep_cfg = sweep_cfgs[pidx]
    act_queue = act_queues[pidx]

    if cfg.act_norms is not None:
        # multi-layer autoencoder
        encoder_cfg = AutoEncoderMultiLayerConfig(
            n_dim=cfg.n_dim,
            m_dim=cfg.m_dim,
            act_norms=cfg.act_norms,
            act_renorm_type=sweep_cfg["act_renorm_type"],
            act_renorm_scale=sweep_cfg["act_renorm_scale"],
            device=cfg.device,
            dtype=cfg.dtype,
            lambda_reg=sweep_cfg["lambda_reg"],
            record_data=True,
        )

        trainer_cfg = AutoEncoderMultiLayerTrainerConfig(
            lr=sweep_cfg["lr"],
            beta1=sweep_cfg["beta1"],
            beta2=sweep_cfg["beta2"],
            total_steps=cfg.total_activations // cfg.batch_size,
            warmup_percent=sweep_cfg["warmup_percent"],
            wb_project=cfg.wb_project,
            wb_entity=cfg.wb_entity,
            wb_name="{}: ML_R{:.1e}_rt{}_rs{:g}_LR={:.1e}".format(
                offset + pidx,
                sweep_cfg["lambda_reg"],
                sweep_cfg["act_renorm_type"],
                sweep_cfg["act_renorm_scale"],
                sweep_cfg["lr"],
                ),
            wb_group=cfg.wb_group,
            wb_config={
                **(cfg.wb_config or {}),
                **sweep_cfg,
            },
        )

        trainer = AutoEncoderMultiLayerTrainer(encoder_cfg, trainer_cfg)
    else:
        # single-layer autoencoder
        encoder_cfg = AutoEncoderConfig(
            n_dim=cfg.n_dim,
            m_dim=cfg.m_dim,
            device=cfg.device,
            dtype=cfg.dtype,
            lambda_reg=sweep_cfg["lambda_reg"],
            record_data=True,
        )

        trainer_cfg = AutoEncoderTrainerConfig(
            lr=sweep_cfg["lr"],
            beta1=sweep_cfg["beta1"],
            beta2=sweep_cfg["beta2"],
            total_steps=cfg.total_activations // cfg.batch_size,
            warmup_percent=sweep_cfg["warmup_percent"],
            wb_project=cfg.wb_project,
            wb_entity=cfg.wb_entity,
            wb_name="{}: L{}_R{:.1e}_LR={:.1e}".format(
                offset + pidx,
                sweep_cfg["layer"],
                sweep_cfg["lambda_reg"],
                sweep_cfg["lr"],
                sweep_cfg["beta1"],
                sweep_cfg["beta2"],
                sweep_cfg["warmup_percent"],
                ),
            wb_group=cfg.wb_group,
            wb_config={
                **(cfg.wb_config or {}),
                **sweep_cfg,
            },
        )

        trainer = AutoEncoderTrainer(encoder_cfg, trainer_cfg)

    try:
        while True:
            acts = act_queue.get(block=True, timeout=None)

            if acts is None:
                break

            trainer.train_on(acts)
            del acts
            act_queue.task_done()
    except Exception as _:
        logging.error(f"Process #{pidx} encountered an error:\n{traceback.format_exc()}")
        # re-raise the exception in the main process
        sys.exit(1)
    finally:
        act_queue.task_done()
        trainer.finish()


class AutoEncoderSweeper:
    def __init__(self, cfg: AutoEncoderSweeperConfig, buffer: ActivationsBuffer):
        self.cfg = cfg
        self.buffer = buffer

        if cfg.act_norms is not None:
            # multi-layer autoencoder sweep configs
            self.sweep_cfgs = [
                {
                    "lr": lr,
                    "beta1": beta1,
                    "beta2": beta2,
                    "lambda_reg": lambda_reg,
                    "warmup_percent": warmup_percent,
                    "act_renorm_type": act_renorm_type,
                    "act_renorm_scale": act_renorm_scale,
                }
                for lr in cfg.lr
                for beta1 in cfg.beta1
                for beta2 in cfg.beta2
                for lambda_reg in cfg.lambda_reg
                for warmup_percent in cfg.warmup_percent
                for act_renorm_type in cfg.act_renorm_type
                for act_renorm_scale in cfg.act_renorm_scale
            ]

        else:
            # single-layer autoencoder sweep configs
            self.sweep_cfgs = [
                {
                    "lr": lr,
                    "beta1": beta1,
                    "beta2": beta2,
                    "lambda_reg": lambda_reg,
                    "warmup_percent": warmup_percent,
                    "layer": layer,
                }
                for lr in cfg.lr
                for beta1 in cfg.beta1
                for beta2 in cfg.beta2
                for lambda_reg in cfg.lambda_reg
                for warmup_percent in cfg.warmup_percent
                for layer in cfg.layer  # layer is last so that it is always iterated over in adjacent cfgs, this speeds
                # up the sweep since the full set activations from the model can always be used
            ]

    def run(self):
        print(f"Running sweep with {len(self.sweep_cfgs)} configurations")

        # required for torch.multiprocessing to work with CUDA tensors
        torch.multiprocessing.set_start_method('spawn', force=True)

        multiprocessing.log_to_stderr(logging.ERROR)

        for i in range(0, len(self.sweep_cfgs), self.cfg.parallelism):
            num_trainers = min(self.cfg.parallelism, len(self.sweep_cfgs) - i)

            queues = [Queue(maxsize=1) for _ in range(num_trainers)]

            print(f"Running configs {i + 1} to {i + num_trainers} of {len(self.sweep_cfgs)}")

            # reset buffer
            self.buffer.reset_dataset()

            active_sweep_cfgs = self.sweep_cfgs[i:i + self.cfg.parallelism]

            trainer_workers = spawn(
                create_trainer_worker,
                nprocs=num_trainers,
                args=(i + 1, active_sweep_cfgs, queues, self.cfg),
                join=False
            )

            for _ in tqdm(range(self.cfg.total_activations // self.cfg.batch_size)):
                acts = self.buffer.next(batch=self.cfg.batch_size)
                acts = acts.to(self.cfg.device, dtype=self.cfg.dtype)

                # block until all previous activations have been processed
                # this comes before so that buffer.next() can be called in parallel, since it can be slow to refresh
                for q in queues:
                    q.join()

                for j, q in enumerate(queues):
                    if self.cfg.act_norms is not None:
                        q.put(acts)
                    else:
                        q.put(acts[:, active_sweep_cfgs[j]["layer"], :])

            for q in queues:
                q.put(None)

            trainer_workers.join()

            for q in queues:
                q.close()

            gc.collect()
            torch.cuda.empty_cache()
