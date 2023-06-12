import torch.nn as nn
from utils import *
from adapter.adapting import AsmdAdapter

def replace_lang_emb(model, shared_code=None, shared_text=None, only_src=False):        
    src_word_emb = nn.Embedding(src_vocab_size, d_word_vec, padding_idx=src_pad_idx).to(device)
    trg_word_emb = nn.Embedding(trg_vocab_size, d_word_vec, padding_idx=trg_pad_idx).to(device)
    trg_word_prj = nn.Linear(d_model, trg_vocab_size, bias=False).to(device)    
    
    if shared_code:
       src_word_emb.weight.data[:len(shared_code)] = model.module.encoder.src_word_emb.weight.data[shared_code]
    
    if shared_text and not only_src:        
        trg_word_emb.weight.data[:len(shared_text)] = model.module.decoder.trg_word_emb.weight.data[shared_text]

    model.module.encoder.src_word_emb = src_word_emb    
    if not only_src:
        model.module.decoder.trg_word_emb = trg_word_emb
        model.module.trg_word_prj = trg_word_prj    
    
    model = AsmdAdapter(model, only_src=False)
    for name, param in model.named_parameters():
        if 'encoder.src_word_emb' in name or ('decoder.trg_word_emb' in name \
            or 'module.trg_word_prj' in name if not only_src else False):            
            param.requires_grad = True          
        
 

    train_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print("Trainable params: ", train_params,
          total_params, train_params/total_params)


if __name__ == '__main__':
    """to do"""