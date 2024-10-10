import os
from typing import Optional
from torch.utils.tensorboard import SummaryWriter
import socket
from datetime import datetime
from functools import cached_property


RESOURCES_DIR = os.path.join(os.path.dirname(__file__), "resources")
SPM_VOCAB_PATH = os.path.join(RESOURCES_DIR, "spm.vocab")
SPM_MODEL_PATH = os.path.join(RESOURCES_DIR, "spm.model")

CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")
GUACAMOL_DIR = os.path.join(CACHE_DIR, "guacamol")

class TrainingCfg:
    LR_WARMUP = True
    LR_BATCH_SCALING = True

class EngineCfg:
    NUM_WORKERS = 4
    device = "cpu"

class _RunCfg:
    @classmethod
    def RESUME(cls) -> bool:
        return bool(int(os.environ.get("RESUME", "0")))
    @classmethod
    def EXP_NAME(cls) -> str:
        return os.environ.get("EXP_NAME", "_tmp")
    
    @cached_property
    def log_dir(self) -> str:
        exp_dir = os.path.join("runs", self.EXP_NAME())
        if not self.RESUME():
            ret = os.path.join(exp_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + socket.gethostname())
            os.makedirs(ret)
        
        last_run_name = sorted(os.listdir(exp_dir))[-1]
        return os.path.join(exp_dir, last_run_name)
    
    @cached_property
    def writer(self) -> SummaryWriter:
        return SummaryWriter(self.log_dir)
        
    def LAST_CKPT_PATH(self) -> str:
     return os.path.join(self.log_dir, "last_ckpt.pt")

    def BEST_CKPT_PATH(self, metric_name: str) -> str:
        return os.path.join(self.log_dir, f"best_{metric_name}_ckpt.pt")
    
    global_step = 0
RunCfg = _RunCfg() # type: ignore