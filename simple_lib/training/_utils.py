from typing import Literal, Tuple
from dataclasses import dataclass, field
import signal
import logging
from collections import defaultdict
import os
import math

import simple_lib
from simple_lib.logging.logging import MetricLogger, logger

import torch
from torch import nn
from torch.utils.data import DataLoader, BatchSampler, RandomSampler, SequentialSampler, Dataset
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, LambdaLR, ChainedScheduler


def get_default_batch_samplers(train_dataset: Dataset, valid_dataset: Dataset, batch_dim: int) -> Tuple[BatchSampler, BatchSampler]:
    train_batch_sampler = BatchSampler(RandomSampler(train_dataset), batch_dim, drop_last=True) # type: ignore
    valid_batch_sampler = BatchSampler(SequentialSampler(valid_dataset), batch_dim, drop_last=False) # type: ignore
    return train_batch_sampler, valid_batch_sampler

class _DelayedInterrupts:
    def __enter__(self):
        self._signals_received = []
        self._old_handlers = {sig: signal.signal(sig, self.handler) for sig in (signal.SIGINT, signal.SIGTERM)}
                
    def handler(self, sig, frame):
        self._signals_received.append((sig, frame))
        logging.warning('Graceful shutdown, delaying KeyboardInterrupt.')
    
    def __exit__(self, type, value, traceback):
        for sig, handler in self._old_handlers.items():
            signal.signal(sig, handler)
        for sig, frame in self._signals_received:
            self._old_handlers[sig](sig, frame)

@dataclass
class TrainState:
    criterion: nn.Module
    optimizer: Optimizer
    lr_scheduler: LRScheduler
    epoch: int
    min_metrics: defaultdict[str, float] = field(default_factory=lambda: defaultdict(lambda: math.inf))

    def save_state_dicts(self, path: str):
        torch.save(
            {"module": self.criterion.state_dict(),
             "optimizer": self.optimizer.state_dict(),
             "lr_scheduler": self.lr_scheduler.state_dict(),
             "epoch": self.epoch,
             "min_metrics": dict(self.min_metrics)},
            path)

    def load_state_dicts(self, path: str):
        state = torch.load(path, map_location=simple_lib.EngineCfg.device, weights_only=True)
        self.criterion.load_state_dict(state["module"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.lr_scheduler.load_state_dict(state["lr_scheduler"])
        self.epoch = state["epoch"]
        for k, v in state["min_metrics"].items():
            self.min_metrics[k] = v

def step(dataloader: DataLoader,
         state: TrainState,
         mode: Literal["train", "valid"],
         header: str = ""):
    if mode == "train":
        state.criterion.train()
    elif mode == "valid":
        state.criterion.eval()
    else:
        assert False
    
    do_warmup = mode == "train" and simple_lib.TrainingCfg.LR_WARMUP and state.epoch == 0
    if do_warmup:
        warmup_lr_scheduler = LambdaLR(state.optimizer, lambda it: it/max(len(dataloader), 1))

    metric_logger = MetricLogger()
    
    for batch in metric_logger.log_every(dataloader, 10, f"Mode: {mode}, " + header):
        assert isinstance(batch, tuple), f"Expected tuple, got {type(batch)}"
        assert isinstance(batch[0], torch.Tensor), f"Expected first element in batch to be tensor, got {type(batch[0])}"
        batch = type(batch)(
            *[x.to(simple_lib.EngineCfg.device, non_blocking=True) if isinstance(x, torch.Tensor) else x
              for x in batch])
        n_samples = len(batch[0])
        
        with torch.set_grad_enabled(mode=="train"):
            metrics = state.criterion(batch)
        
        if mode == "train":
            simple_lib.RunCfg.global_step += 1
            state.optimizer.zero_grad()
            metrics["loss"].backward()
            state.optimizer.step()

            for i, param_grp in enumerate(state.optimizer.param_groups):
                simple_lib.RunCfg.writer.add_scalar(f"{mode}/lr_{i}", param_grp["lr"], global_step=simple_lib.RunCfg.global_step)
            simple_lib.RunCfg.writer.add_scalar(f"{mode}/batch_loss", metrics["loss"].item(), global_step=simple_lib.RunCfg.global_step)
            
        if do_warmup:
            warmup_lr_scheduler.step()
        metric_logger.update(n_samples, **metrics)
        
    if mode == "train" and not do_warmup:
        state.lr_scheduler.step()
    
    global_metrics = {k: v.global_avg for k, v in metric_logger.meters.items()}
    logger.info(str(global_metrics))
    with _DelayedInterrupts():
        assert simple_lib.RunCfg.writer is not None
        for k, v in global_metrics.items():
            simple_lib.RunCfg.writer.add_scalar(f"{mode}/{k}", v, global_step=simple_lib.RunCfg.global_step)
        if mode == "valid":
            for metric_name, v in global_metrics.items():
                min_v = state.min_metrics[metric_name]
                if v < min_v:
                    torch.save(state.criterion.state_dict(), simple_lib.RunCfg.BEST_CKPT_PATH(metric_name))
                state.min_metrics[metric_name] = min(min_v, v)
            state.save_state_dicts(simple_lib.RunCfg.LAST_CKPT_PATH())

    return global_metrics

def handle_lr_batch_scaling(avg_batch_size: int, optimizer: Optimizer, lr_scheduler: LRScheduler) -> LRScheduler:
    if simple_lib.TrainingCfg.LR_BATCH_SCALING:
        return ChainedScheduler(
            [LambdaLR(optimizer, lambda _: avg_batch_size / 32), lr_scheduler],
            optimizer)
    else:
        return lr_scheduler

def handle_resume(criterion: nn.Module, optimizer: Optimizer, lr_scheduler: LRScheduler) -> TrainState:
    ret = TrainState(criterion, optimizer, lr_scheduler, -1)
    if simple_lib.RunCfg.RESUME():
        if os.path.isfile(simple_lib.RunCfg.LAST_CKPT_PATH()):
            ret.load_state_dicts(simple_lib.RunCfg.LAST_CKPT_PATH())
    return ret