# Copied from https://github.com/3DTopia/OpenLRM/blob/main/openlrm/runners/train/base_trainer.py

import os
import time
import math
import argparse
import shutil
import torch
import safetensors
from omegaconf import OmegaConf
from abc import abstractmethod
from contextlib import contextmanager
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from FastAvatar.utils.logging import configure_logger
from FastAvatar.utils.compile import configure_dynamo
from FastAvatar.runners.abstract import Runner


logger = get_logger(__name__)


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/train.yaml')
    args, unknown = parser.parse_known_args()

    # Load base configuration
    cfg = OmegaConf.load(args.config)

    # Override with command-line arguments
    cli_cfg = OmegaConf.from_cli(unknown)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    return cfg


class Trainer(Runner):

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")

        # Initialize accelerator
        self.accelerator = Accelerator(
            gradient_accumulation_steps=self.cfg.train.accum_steps,
            log_with=tuple(self.cfg.logger.trackers) if hasattr(self.cfg.logger, 'trackers') else None,
            project_config=ProjectConfiguration(
                logging_dir=self.cfg.logger.tracker_root if hasattr(self.cfg.logger, 'tracker_root') else './logs',
            ),
            kwargs_handlers=[
                DistributedDataParallelKwargs(
                    find_unused_parameters=self.cfg.train.find_unused_parameters,
                ),
            ],
        )

        # Set random seed
        set_seed(self.cfg.experiment.seed if hasattr(self.cfg.experiment, 'seed') else 42, device_specific=True)

        # Configure logging
        with self.accelerator.main_process_first():
            configure_logger(
                stream_level=self.cfg.logger.stream_level if hasattr(self.cfg.logger, 'stream_level') else 'INFO',
                log_level=self.cfg.logger.log_level if hasattr(self.cfg.logger, 'log_level') else 'INFO',
                file_path=os.path.join(
                    self.cfg.logger.log_root if hasattr(self.cfg.logger, 'log_root') else './logs',
                    self.cfg.experiment.parent if hasattr(self.cfg.experiment, 'parent') else 'default',
                    self.cfg.experiment.child if hasattr(self.cfg.experiment, 'child') else 'default',
                    f"{self.timestamp}.log",
                ) if self.accelerator.is_main_process else None,
            )

        # Configure torch dynamo if specified
        if hasattr(self.cfg, 'compile'):
            configure_dynamo(dict(self.cfg.compile))

        # Initialize core components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.N_max_global_steps = None
        self.N_global_steps_per_epoch = None
        self.global_step = 0
        self.current_epoch = 0

    def __enter__(self):
        self.accelerator.init_trackers(
            project_name=f"{self.cfg.experiment.parent if hasattr(self.cfg.experiment, 'parent') else 'default'}/{self.cfg.experiment.child if hasattr(self.cfg.experiment, 'child') else 'default'}",
        )
        self.prepare_everything()
        self.log_initial_info()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.accelerator.end_training()

    @staticmethod
    def control(option: str = None, synchronized: bool = False):
        def decorator(func):
            def wrapper(self, *args, **kwargs):
                if option is None or hasattr(self.accelerator, option):
                    accelerated_func = getattr(self.accelerator, option)(func) if option is not None else func
                    result = accelerated_func(self, *args, **kwargs)
                    if synchronized:
                        self.accelerator.wait_for_everyone()
                    return result
                else:
                    raise AttributeError(f"Accelerator has no attribute {option}")
            return wrapper
        return decorator

    @contextmanager
    def exec_in_order(self):
        for rank in range(self.accelerator.num_processes):
            try:
                if self.accelerator.process_index == rank:
                    yield
            finally:
                self.accelerator.wait_for_everyone()

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self) -> bool:
        return self.accelerator.num_processes > 1

    def prepare_everything(self, is_dist_validation: bool = True):
        if self.is_distributed and not torch.distributed.is_initialized():
            logger.info("Initializing distributed process group...")
            torch.distributed.init_process_group(backend='nccl')
            logger.info(f"Distributed process group initialized. Rank: {torch.distributed.get_rank()}, World Size: {torch.distributed.get_world_size()}")
        
        # Prepare with accelerator
        if is_dist_validation:
            self.model, self.optimizer, self.train_loader, self.val_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.val_loader,
                )
        else:
            self.model, self.optimizer, self.train_loader = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader,
                )

        # Register scheduler for checkpointing
        if self.scheduler is not None:
            if isinstance(self.scheduler, dict):
                for name, sched in self.scheduler.items():
                    self.accelerator.register_for_checkpointing(sched)
            else:
                self.accelerator.register_for_checkpointing(self.scheduler)

        # Calculate training statistics
        N_total_batch_size = self.cfg.train.batch_size * self.accelerator.num_processes * self.cfg.train.accum_steps
        self.N_global_steps_per_epoch = math.ceil(len(self.train_loader) / self.cfg.train.accum_steps)
        self.N_max_global_steps = self.N_global_steps_per_epoch * self.cfg.train.epochs

        # Override max steps if debug mode is enabled
        if hasattr(self.cfg.train, 'debug_global_steps') and self.cfg.train.debug_global_steps is not None:
            logger.warning(f"Overriding max global steps from {self.N_max_global_steps} to {self.cfg.train.debug_global_steps}")
            self.N_max_global_steps = self.cfg.train.debug_global_steps

        # Log training statistics
        logger.info(f"======== Training Statistics ========")
        logger.info(f"** Max global steps: {self.N_max_global_steps}")
        logger.info(f"** Total batch size: {N_total_batch_size}")
        logger.info(f"** Number of epochs: {self.cfg.train.epochs}")
        logger.info(f"** Global steps per epoch: {self.N_global_steps_per_epoch}")
        logger.info(f"** Distributed validation: {is_dist_validation}")
        logger.info(f"====================================")

        # Log model parameters
        logger.info(f"======== Model Parameters ========")
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"** Total trainable parameters: {total_params:,}")
        for name, module in self.accelerator.unwrap_model(self.model).named_children():
            trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            if trainable_params > 0:
                logger.info(f"** {name}: {trainable_params:,} trainable parameters")
        logger.info(f"=================================")

        self.accelerator.wait_for_everyone()

        # Load checkpoint or model
        self.load_ckpt_or_auto_resume_(self.cfg)

        # Register hooks
        self.register_hooks()

    @abstractmethod
    def register_hooks(self):
        pass

    def auto_resume_(self, cfg) -> bool:
        ckpt_root = os.path.join(
            cfg.saver.checkpoint_root if hasattr(cfg.saver, 'checkpoint_root') else './checkpoints',
            cfg.experiment.parent if hasattr(cfg.experiment, 'parent') else 'default',
            cfg.experiment.child if hasattr(cfg.experiment, 'child') else 'default',
        )
        
        if not os.path.exists(ckpt_root):
            return False
            
        ckpt_dirs = os.listdir(ckpt_root)
        if len(ckpt_dirs) == 0:
            return False
            
        ckpt_dirs.sort()
        latest_ckpt = ckpt_dirs[-1]
        latest_ckpt_dir = os.path.join(ckpt_root, latest_ckpt)
        
        logger.info(f"======== Auto-resuming from {latest_ckpt_dir} ========")
        self.accelerator.load_state(latest_ckpt_dir)
        self.global_step = int(latest_ckpt)
        self.current_epoch = self.global_step // self.N_global_steps_per_epoch
        return True

    def load_model_(self, cfg):
        if not hasattr(cfg.saver, 'load_model') or not cfg.saver.load_model:
            return False
            
        logger.info(f"======== Loading model from {cfg.saver.load_model} ========")
        safetensors.torch.load_model(
            self.accelerator.unwrap_model(self.model),
            cfg.saver.load_model,
            strict=True,
        )
        logger.info(f"======== Model loaded successfully ========")
        return True

    @control(synchronized=True)
    def load_ckpt_or_auto_resume_(self, cfg):
        if hasattr(cfg.saver, 'auto_resume') and cfg.saver.auto_resume:
            if self.auto_resume_(cfg):
                return
                
        if hasattr(cfg.saver, 'load_model') and cfg.saver.load_model:
            if self.load_model_(cfg):
                return
                
        logger.debug("======== No checkpoint or model loaded ========")

    @control('on_main_process', synchronized=True)
    def save_checkpoint(self):
        ckpt_dir = os.path.join(
            self.cfg.saver.checkpoint_root if hasattr(self.cfg.saver, 'checkpoint_root') else './checkpoints',
            self.cfg.experiment.parent if hasattr(self.cfg.experiment, 'parent') else 'default',
            self.cfg.experiment.child if hasattr(self.cfg.experiment, 'child') else 'default',
            f"{self.global_step:06d}",
        )
        
        self.accelerator.save_state(output_dir=ckpt_dir, safe_serialization=True)
        logger.info(f"======== Saved checkpoint at global step {self.global_step} ========")

        # Manage checkpoints based on retention policy
        if hasattr(self.cfg.saver, 'checkpoint_keep_level') and hasattr(self.cfg.saver, 'checkpoint_global_steps'):
            ckpt_dirs = os.listdir(os.path.dirname(ckpt_dir))
            ckpt_dirs.sort()
            max_ckpt = int(ckpt_dirs[-1])
            ckpt_base = int(self.cfg.saver.checkpoint_keep_level)
            ckpt_period = self.cfg.saver.checkpoint_global_steps
            
            cur_order = ckpt_base ** math.floor(math.log(max_ckpt // ckpt_period, ckpt_base))
            cur_idx = 0
            
            while cur_order > 0:
                cur_digit = max_ckpt // ckpt_period // cur_order % ckpt_base
                while cur_idx < len(ckpt_dirs) and int(ckpt_dirs[cur_idx]) // ckpt_period // cur_order % ckpt_base < cur_digit:
                    if int(ckpt_dirs[cur_idx]) // ckpt_period % cur_order != 0:
                        shutil.rmtree(os.path.join(os.path.dirname(ckpt_dir), ckpt_dirs[cur_idx]))
                        logger.info(f"Removed checkpoint {ckpt_dirs[cur_idx]}")
                    cur_idx += 1
                cur_order //= ckpt_base

    @property
    def global_step_in_epoch(self):
        return self.global_step % self.N_global_steps_per_epoch

    @abstractmethod
    def _build_model(self):
        pass

    @abstractmethod
    def _build_optimizer(self):
        pass

    @abstractmethod
    def _build_scheduler(self):
        pass

    @abstractmethod
    def _build_dataloader(self):
        pass

    @abstractmethod
    def _build_loss_fn(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    @staticmethod
    def _get_str_progress(epoch: int = None, step: int = None):
        if epoch is not None:
            log_type = 'epoch'
            log_progress = epoch
        elif step is not None:
            log_type = 'step'
            log_progress = step
        else:
            raise ValueError('Either epoch or step must be provided')
        return log_type, log_progress

    @control('on_main_process')
    def log_scalar_kwargs(self, epoch: int = None, step: int = None, split: str = None, **scalar_kwargs):
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f'/{split}' if split else ''
        for key, value in scalar_kwargs.items():
            self.accelerator.log({f'{key}{split}/{log_type}': value}, log_progress)

    @control('on_main_process')
    def log_images(self, values: dict, step: int | None = None, log_kwargs: dict | None = {}):
        for tracker in self.accelerator.trackers:
            if hasattr(tracker, 'log_images'):
                tracker.log_images(values, step=step, **log_kwargs.get(tracker.name, {}))

    @control('on_main_process')
    def log_optimizer(self, epoch: int = None, step: int = None, attrs: list[str] = [], group_ids: list[int] = []):
        log_type, log_progress = self._get_str_progress(epoch, step)
        assert self.optimizer is not None, 'Optimizer is not initialized'
        if not attrs:
            logger.warning('No optimizer attributes are provided, nothing will be logged')
        if not group_ids:
            logger.warning('No optimizer group ids are provided, nothing will be logged')
        for attr in attrs:
            assert attr in ['lr', 'momentum', 'weight_decay'], f'Invalid optimizer attribute {attr}'
            for group_id in group_ids:
                self.accelerator.log({f'opt/{attr}/{group_id}': self.optimizer.param_groups[group_id][attr]}, log_progress)

    @control('on_main_process')
    def log_initial_info(self):
        assert self.model is not None, 'Model is not initialized'
        assert self.optimizer is not None, 'Optimizer is not initialized'
        assert self.scheduler is not None, 'Scheduler is not initialized'
        
        # Log configuration
        self.accelerator.log({'Config': "```\n" + OmegaConf.to_yaml(self.cfg) + "\n```"})
        
        # Log model architecture
        self.accelerator.log({'Model': "```\n" + str(self.model) + "\n```"})
        
        # Log optimizer configuration
        self.accelerator.log({'Optimizer': "```\n" + str(self.optimizer) + "\n```"})
        
        # Log scheduler configuration
        self.accelerator.log({'Scheduler': "```\n" + str(self.scheduler) + "\n```"})

    def run(self):
        self.train()