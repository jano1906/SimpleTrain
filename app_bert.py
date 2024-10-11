from simple_lib.training.train_bert import train_bert
from simple_lib.training.train_bert import Bert
from simple_lib.data.data import load_tokenized_data
from simple_lib.nn.models import TransformerEncoder
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from simple_lib.logging.logging import setup_logging
import simple_lib

from torch import nn

class MyBert(Bert):
    def __init__(self, encoder: TransformerEncoder, **kwargs):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.args.embedding_dim, encoder.args.vocab_size)
        self.kwargs = kwargs
    
    def special_tokens(self) -> dict:
        return self.kwargs
    
    def predict(self, masked_encoder_in):
        enc = self.encoder.encode(masked_encoder_in, pad_id=self.pad_id())
        pred = self.head(enc)
        return pred
    
if __name__ == "__main__":
    N_EPOCHS = 20
    BATCH_DIM = 32

    data, stats, tokenizer = load_tokenized_data("guacamol", "sentencepiece")
    # for test, shorten the data
    data.train_split = data.train_split[:1024]  # type: ignore
    data.valid_split = data.valid_split[:1024]  # type: ignore 
    data.test_split = data.test_split[:1024]    # type: ignore
    
    model = TransformerEncoder(TransformerEncoder.Args(tokenizer.max_id+4, 128, 4, 4))
    optimizer = AdamW(model.parameters())
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-8)

    simple_lib.EngineCfg.device = "cuda"
    simple_lib.EngineCfg.NUM_WORKERS = 0

    simple_lib.TrainingCfg.LR_WARMUP = True
    simple_lib.TrainingCfg.LR_BATCH_SCALING = True

    setup_logging()
    
    train_bert(
        MyBert(model, cls_ids=[tokenizer.max_id+1], pad_id=tokenizer.max_id+2, mask_id=tokenizer.max_id+3),
        data,
        stats,
        optimizer,
        lr_scheduler,
        BATCH_DIM,
        N_EPOCHS,
    )