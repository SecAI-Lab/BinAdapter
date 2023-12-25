from torchtext.data import Field, BucketIterator, TabularDataset
from model.asmdepictor.Optim import ScheduledOptim
import sys

from utils import *
from config import *
from voca import AssemblyVoca
from train import trainIters

if __name__ == "__main__":
    pretrained_path = sys.argv[-1]
    print(pretrained_path)

    model = load_model(pretrained_path)
    model = replace_lang_emb(model)
    voca = AssemblyVoca(text_path=text_voca_path, code_path=code_voca_path)

    code = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenize,
        lower=True,
        pad_token="<pad>",
        fix_length=max_token_seq_len,
    )

    text = Field(
        sequential=True,
        use_vocab=True,
        tokenize=tokenize,
        lower=True,
        init_token="<sos>",
        eos_token="<eos>",
        pad_token="<pad>",
        fix_length=max_token_seq_len,
    )

    fields = {"Code": ("code", code), "Text": ("text", text)}

    train_data, valid_data, test_data = TabularDataset.splits(
        path="",
        train=train_json,
        test=test_json,
        validation=test_json,
        format="json",
        fields=fields,
    )

    text, code = voca.build(text, code, train_data)
    voca.save_text(text.vocab)
    voca.save_code(code.vocab)

    src_vocab_size = len(code.vocab.stoi)
    trg_vocab_size = len(text.vocab.stoi)

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=batch_size,
        device="cuda",
        sort=False,
    )

    model_optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul,
        d_model,
        n_warmup_steps,
    )

    trainIters(
        model,
        n_epoch,
        train_iterator,
        valid_iterator,
        model_optimizer,
        smoothing,
        only_src=True,
    )

    """
    model.load_state_dict(torch.load(
        '/path/to/adapter/state/'), strict=False)
    
    model.module.encoder.src_word_emb = torch.load(
        '/path/to/src_embedding/')
    
    """
