import simple_lib
from simple_lib.data._utils import TokenizedData, TokenizedDataStats
from simple_lib.training._utils import step, handle_lr_batch_scaling, get_default_batch_samplers, handle_resume

import torch
from torch import nn
import torch.nn.functional as F

from typing import NamedTuple, Protocol
import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import Dataset, DataLoader

class BertProtocol(Protocol):
    def special_tokens(self) -> dict: pass
    def pad_id(self) -> int: 
        return self.special_tokens()["pad_id"]
    def mask_id(self) -> int: 
        return self.special_tokens()["mask_id"]
    def cls_ids(self) -> list[int]: 
        return self.special_tokens()["cls_ids"]

    def predict(self, masked_encoder_in: torch.Tensor) -> torch.Tensor: pass    
    
class Bert(nn.Module, BertProtocol): pass

class BertSample(NamedTuple):
    masked_encoder_in: torch.Tensor
    masked_idx: torch.Tensor
    target: torch.Tensor

class BertDataset(Dataset[BertSample]):
    def __init__(self, tokenized_data: Dataset[list[int]], cls_ids: list[int], mask_id: int, pad_id: int):
        self.tokenized_data = tokenized_data
        self.cls_ids = [id for id in cls_ids]
        self.mask_id = mask_id
        self.pad_id = pad_id
        
    def __len__(self):
        return len(self.tokenized_data)

    def __getitem__(self, idx: int) -> BertSample:
        sample = torch.tensor(self.tokenized_data[idx])
        n = len(sample)
        n_masked = math.ceil(n * 0.15)
        masked_idx = torch.multinomial(torch.ones([n]), n_masked)
        target = sample[masked_idx]
        sample[masked_idx] = self.mask_id
        masked_encoder_in = torch.concat([torch.tensor(self.cls_ids), sample])
        masked_idx += len(self.cls_ids)

        return BertSample(
            masked_encoder_in = masked_encoder_in,
            masked_idx = masked_idx,
            target = target,
        )
    
    def get_collate_fn(self, stats: TokenizedDataStats):
        max_input_len = stats.max_sample_len + len(self.cls_ids)
        def collate(batch: list[BertSample]):
            return BertSample(
                masked_encoder_in=torch.stack([F.pad(x.masked_encoder_in, (0, max_input_len-len(x.masked_encoder_in)), value=self.pad_id) for x in batch]),
                masked_idx=torch.concat([x.masked_idx + i*max_input_len for i, x in enumerate(batch)]),
                target=torch.concat([x.target for x in batch])
            )
        return collate


class BertLoss(nn.Module):
    def __init__(self, model: Bert):
        super().__init__()
        self.model = model

    def forward(self, sample: BertSample):
        pred = self.model.predict(sample.masked_encoder_in).flatten(0, 1)
        out = pred[sample.masked_idx]
        loss = F.cross_entropy(out, sample.target, ignore_index=self.model.pad_id())
        return {"loss": loss}

def train_bert(model: Bert,
                tokenized_data: TokenizedData,
                data_stats: TokenizedDataStats,
                optimizer: Optimizer,
                lr_scheduler: LRScheduler,
                batch_dim: int,
                n_epochs: int):

    train_dataset = BertDataset(tokenized_data=tokenized_data.train_split, cls_ids=model.cls_ids(), pad_id=model.pad_id(), mask_id=model.mask_id())
    valid_dataset = BertDataset(tokenized_data=tokenized_data.valid_split, cls_ids=model.cls_ids(), pad_id=model.pad_id(), mask_id=model.mask_id())
    train_batch_sampler, valid_batch_sampler = get_default_batch_samplers(train_dataset, valid_dataset, batch_dim)
    criterion = BertLoss(model)
    criterion.to(simple_lib.EngineCfg.device)

    lr_scheduler = handle_lr_batch_scaling(train_batch_sampler.batch_size, optimizer, lr_scheduler)
    state = handle_resume(criterion, optimizer, lr_scheduler)
    
    train_dataloader = DataLoader(train_dataset, collate_fn=train_dataset.get_collate_fn(data_stats), batch_sampler=train_batch_sampler, pin_memory=True, num_workers=simple_lib.EngineCfg.NUM_WORKERS)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=valid_dataset.get_collate_fn(data_stats), batch_sampler=valid_batch_sampler, pin_memory=True, num_workers=simple_lib.EngineCfg.NUM_WORKERS)

    for epoch in range(state.epoch+1, n_epochs):
        state.epoch = epoch
        train_metrics = step(train_dataloader, state, "train", f"Epoch: {epoch+1}/{n_epochs}")
        valid_metrics = step(valid_dataloader, state, "valid", f"Epoch: {epoch+1}/{n_epochs}")
