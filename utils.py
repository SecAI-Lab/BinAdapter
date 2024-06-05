from torchtext.data import Field, BucketIterator, TabularDataset
from model.asmdepictor.Models import Asmdepictor
from model.asmdepictor.Optim import ScheduledOptim
from sklearn.utils import shuffle
import torch.nn as nn
import pandas as pd
import torch
from collections import OrderedDict
from config import *


def tokenize(x):
    return x.split()


def load_model(model_path, params):
    print("Loading ", model_path)
    try:
        model = torch.load(model_path)
    except:
        print("File not found or not loadable!")
        exit(0)

    model = Asmdepictor(
        params["src_vocab_size"],
        33546,  # should be fixed size of pretrained model trg_vocab_size
        src_pad_idx=params["src_pad_idx"],
        trg_pad_idx=params["trg_pad_idx"],
        trg_emb_prj_weight_sharing=proj_share_weight,
        emb_src_trg_weight_sharing=embs_share_weight,
        d_k=d_k,
        d_v=d_v,
        d_model=d_model,
        d_word_vec=d_word_vec,
        d_inner=d_inner_hid,
        n_layers=n_layers,
        n_head=n_head,
        dropout=dropout,
        scale_emb_or_prj=scale_emb_or_prj,
        n_position=max_token_seq_len + 3,
    ).to(device)

    if isinstance(model, OrderedDict):
        model.load_state_dict(model_path)
    model = nn.DataParallel(model)
    return model


def preprocessing(src_file, tgt_file, max_token_seq_len, json_file):
    src_data = open(src_file, encoding="utf-8").read().split("\n")
    tgt_data = open(tgt_file, encoding="utf-8").read().split("\n")
    src_text_tok = [line.split() for line in src_data]
    src_tok_concat = [" ".join(tok[0:max_token_seq_len]) for tok in src_text_tok]

    tgt_text_tok = [line.split() for line in tgt_data]
    tgt_tok_concat = [" ".join(tok[0:max_token_seq_len]) for tok in tgt_text_tok]

    raw_data = {
        "Code": [line for line in src_tok_concat],
        "Text": [line for line in tgt_tok_concat],
    }

    df = pd.DataFrame(raw_data, columns=["Code", "Text"])
    df = shuffle(df)

    return df.to_json(json_file, orient="records", lines=True)


def replace_lang_emb(model, params, shared_code=None, shared_text=None, only_src=False):
    src_word_emb = nn.Embedding(
        params["src_vocab_size"], d_word_vec, padding_idx=params["src_pad_idx"]
    ).to(device)
    trg_word_emb = nn.Embedding(
        params["trg_vocab_size"], d_word_vec, padding_idx=params["trg_pad_idx"]
    ).to(device)
    trg_word_prj = nn.Linear(d_model, params["trg_vocab_size"], bias=False).to(device)

    if shared_code:
        src_word_emb.weight.data[: len(shared_code)] = (
            model.module.encoder.src_word_emb.weight.data[shared_code]
        )
        model.module.encoder.src_word_emb = src_word_emb

    if shared_text and not only_src:
        trg_word_emb.weight.data[: len(shared_text)] = (
            model.module.decoder.trg_word_emb.weight.data[shared_text]
        )
        model.module.decoder.trg_word_emb = trg_word_emb
        model.module.trg_word_prj = trg_word_prj


def freeze_params(model, only_src=True):
    for name, param in model.named_parameters():
        if "encoder.src_word_emb" in name or (
            "decoder.trg_word_emb" in name or "module.trg_word_prj" in name
            if not only_src
            else False
        ):
            param.requires_grad = True

    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable params: ", train_params, total_params, train_params / total_params)


def get_fields():
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

    return {"Code": ("code", code), "Text": ("text", text)}


def get_train_test_data():
    train_data, valid_data, test_data = TabularDataset.splits(
        path="",
        train=train_json,
        test=test_json,
        validation=test_json,
        format="json",
        fields=get_fields(),
    )
    return train_data, valid_data, test_data


def get_params(voca, build=False, train_data=None):
    if build and train_data is not None:
        text, code = voca.build(text, code, train_data)
        voca.save_text(text.vocab)
        voca.save_code(code.vocab)
    text_voca = voca.read(text_voca_path)
    code_voca = voca.read(code_voca_path)
    params = {}

    params["src_pad_idx"] = code_voca.stoi["<pad>"]
    params["trg_pad_idx"] = text_voca.stoi["<pad>"]
    params["src_vocab_size"] = len(code_voca.stoi)
    params["trg_vocab_size"] = len(text_voca.stoi)
    return params


def get_optimizer_and_iters(model, train_data, valid_data):
    optimizer = ScheduledOptim(
        torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09),
        lr_mul,
        d_model,
        n_warmup_steps,
    )

    train_iterator, valid_iterator, _ = BucketIterator.splits(
        (train_data, valid_data, valid_data),
        batch_size=batch_size,
        device="cuda",
        sort=False,
    )

    return optimizer, train_iterator, valid_iterator
