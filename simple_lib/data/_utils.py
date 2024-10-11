from dataclasses import dataclass, fields
import multiprocessing as mp
from tqdm import tqdm
from typing import Protocol, Tuple, Sequence, TypeVar, Iterator
import math
from torch.utils.data import Dataset

ItemT = TypeVar("ItemT")
class SeqDataset(Dataset[ItemT]):
    def __init__(self, seq: Sequence[ItemT]):
        self.seq = seq
    
    def __len__(self) -> int:
        return len(self.seq)

    def __getitem__(self, id: int) -> ItemT:
        return self.seq[id]
    
    def __iter__(self) -> Iterator[ItemT]:
        return iter(self.seq)

@dataclass
class RawData:
    train_split: Dataset[str]
    valid_split: Dataset[str]
    test_split: Dataset[str]

@dataclass
class TokenizedData:
    train_split: Dataset[list[int]]
    valid_split: Dataset[list[int]]
    test_split: Dataset[list[int]]

class TokenizerProtocol(Protocol):
    def encode(self, x: str) -> list[int]: pass
    def decode(self, x: list[int]) -> str: pass

class NormalizedTokenizer:
    def __init__(self, tokenizer: TokenizerProtocol, tokenizer_min_id: int, tokenizer_max_id: int, tokenizer_unk_id: int):
        self.tokenizer = tokenizer
        self.tokenizer_min_id = tokenizer_min_id
        self.tokenizer_max_id = tokenizer_max_id
        self.tokenizer_unk_id = tokenizer_unk_id
    @property
    def min_id(self):
        return 0
    @property
    def max_id(self):
        return self.tokenizer_max_id - self.tokenizer_min_id
    @property
    def unk_id(self):
        return self.tokenizer_unk_id - self.tokenizer_min_id
    
    def normalize(self, x: list[int]) -> list[int]:
        return [tok - self.tokenizer_min_id for tok in x]
    
    def unnormalize(self, x: list[int]) -> list[int]:
        return [tok + self.tokenizer_min_id for tok in x]
    
    def encode(self, x: str) -> list[int]:
        return self.normalize(self.tokenizer.encode(x))

    def decode(self, x: list[int]) -> str:
        return self.tokenizer.decode(self.unnormalize(x))


@dataclass
class TokenizedDataStats:
    min_id: int
    max_id: int
    max_sample_len: int

def tokenize_raw_data(raw_data: RawData, tokenizer: TokenizerProtocol, unk_id: int) -> Tuple[TokenizedData, TokenizedDataStats, NormalizedTokenizer]:
    ret = dict()
    
    with mp.Pool(4) as pool:
        for split in fields(raw_data):
            data = getattr(raw_data, split.name)
            ret[split.name] = list(
                tqdm(pool.imap(tokenizer.encode, data, chunksize=1024),
                    desc=f"Tokenizing {split.name}",
                    total=len(data)))
    
    stats = TokenizedDataStats(min_id=unk_id, max_id=unk_id, max_sample_len=-1)
    for data in ret.values():
        split_stats = calculate_tokenized_data_stats(data)
        stats.min_id = min(stats.min_id, split_stats.min_id)
        stats.max_id = max(stats.max_id, split_stats.max_id)
        stats.max_sample_len = max(stats.max_sample_len, split_stats.max_sample_len)
        
    normalized_tokenizer = NormalizedTokenizer(tokenizer, stats.min_id, stats.max_id, unk_id)
    tokenized_data = TokenizedData(**{k: SeqDataset([normalized_tokenizer.normalize(x) for x in v]) for k, v in ret.items()})
    return tokenized_data, stats, normalized_tokenizer


def calculate_tokenized_data_stats(tokenized_data: list[list[int]]) -> TokenizedDataStats:
    min_id = math.inf
    max_id = -math.inf
    max_sample_len = -1
    for sample in tokenized_data:
        min_id = min(min_id, min(sample, default=math.inf))
        max_id = max(max_id, max(sample, default=-math.inf))
        max_sample_len = max(max_sample_len, len(sample))
    return TokenizedDataStats(
        min_id=int(min_id),
        max_id=int(max_id),
        max_sample_len=max_sample_len
    )
