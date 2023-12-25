from model.asmdepictor.Models import Asmdepictor
from sklearn.utils import shuffle
import torch.nn as nn
import pandas as pd
import torch

from adapter.adapting import AsmdAdapter
from config import *


def tokenize(x):
    return x.split()


def load_model(model_path):
    try:
        model = torch.load(model_path)
    except:
        model = Asmdepictor.Asmdepictor(
            src_vocab_size,
            33546,  # trg_vocab_size,
            src_pad_idx=src_pad_idx,
            trg_pad_idx=trg_pad_idx,
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
    return model


def preprocessing(src_file, tgt_file, max_token_seq_len):
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

    return shuffle(df)


def replace_lang_emb(model, shared_code=None, shared_text=None, only_src=False):
    src_word_emb = nn.Embedding(src_vocab_size, d_word_vec, padding_idx=src_pad_idx).to(
        device
    )
    trg_word_emb = nn.Embedding(trg_vocab_size, d_word_vec, padding_idx=trg_pad_idx).to(
        device
    )
    trg_word_prj = nn.Linear(d_model, trg_vocab_size, bias=False).to(device)

    if shared_code:
        src_word_emb.weight.data[
            : len(shared_code)
        ] = model.module.encoder.src_word_emb.weight.data[shared_code]
        model.module.encoder.src_word_emb = src_word_emb

    if shared_text and not only_src:
        trg_word_emb.weight.data[
            : len(shared_text)
        ] = model.module.decoder.trg_word_emb.weight.data[shared_text]
        model.module.decoder.trg_word_emb = trg_word_emb
        model.module.trg_word_prj = trg_word_prj

    # Attaching adapter into model
    model = AsmdAdapter(model)

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
