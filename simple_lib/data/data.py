from typing import Tuple, Callable
import hashlib
import urllib.request
import os
from sentencepiece import SentencePieceProcessor # type: ignore
import simple_lib
from simple_lib.data._utils import SeqDataset, RawData, TokenizedData, TokenizedDataStats, NormalizedTokenizer, tokenize_raw_data


def guacamol_load_raw() -> RawData:
    split_metadata = {
        "train": {
            "link": "https://ndownloader.figshare.com/files/13612760",
            "md5": "05ad85d871958a05c02ab51a4fde8530",
            "local_path": os.path.join(simple_lib.GUACAMOL_DIR, "train.smiles"),
        },
        "valid": {
            "link": "https://ndownloader.figshare.com/files/13612766",
            "md5": "e53db4bff7dc4784123ae6df72e3b1f0",
            "local_path": os.path.join(simple_lib.GUACAMOL_DIR, "valid.smiles"),
        },
        "test": {
            "link": "https://ndownloader.figshare.com/files/13612757",
            "md5": "677b757ccec4809febd83850b43e1616",
            "local_path": os.path.join(simple_lib.GUACAMOL_DIR, "test.smiles"),
        }
    }

    ret = dict()
    
    for split, metadata in split_metadata.items():
        if os.path.isfile(metadata["local_path"]):
            with open(metadata["local_path"]) as f:
                data = f.read().split("\n")
        else:
            with urllib.request.urlopen(metadata["link"]) as f:
                bytes = f.read()
                assert metadata["md5"] == hashlib.md5(bytes).hexdigest()
                data = bytes.decode("utf-8").split("\n")
            os.makedirs(os.path.dirname(metadata["local_path"]), exist_ok=True)
            with open(metadata["local_path"], "w") as f:
                f.write("\n".join(data))
        ret[split] = data
    
    return RawData(
        train_split=SeqDataset(ret["train"]),
        valid_split=SeqDataset(ret["valid"]),
        test_split=SeqDataset(ret["test"]),
    )

RAW_DATA_LOADERS: dict[str, Callable[[], RawData]] = {
    "guacamol": guacamol_load_raw
}

def sentencepiece_tokenize_raw(raw_data: RawData) -> Tuple[TokenizedData, TokenizedDataStats, NormalizedTokenizer]:
    tokenizer = SentencePieceProcessor(simple_lib.SPM_MODEL_PATH)
    return tokenize_raw_data(raw_data=raw_data, tokenizer=tokenizer, unk_id=tokenizer.unk_id())
    
RAW_DATA_TOKENIZERS: dict[str, Callable[[RawData], Tuple[TokenizedData, TokenizedDataStats, NormalizedTokenizer]]] = {
    "sentencepiece": sentencepiece_tokenize_raw,
}

def load_tokenized_data(dataset_name: str, tokenizer_name: str) -> Tuple[TokenizedData, TokenizedDataStats, NormalizedTokenizer]:
    assert dataset_name in RAW_DATA_LOADERS.keys(), f"{dataset_name} not in {RAW_DATA_LOADERS.keys()}"
    assert tokenizer_name in RAW_DATA_TOKENIZERS.keys(), f"{tokenizer_name} not in {RAW_DATA_TOKENIZERS.keys()}"
    load_raw_data = RAW_DATA_LOADERS[dataset_name]
    tokenize_raw_data = RAW_DATA_TOKENIZERS[tokenizer_name]
    
    raw_data = load_raw_data()
    return tokenize_raw_data(raw_data)
