import sys
from adapter.model import BinAdapter
from utils import *
from config import *
from voca import AssemblyVoca


if __name__ == "__main__":
    pretrained_path = sys.argv[-1]  # base model path

    train_set = preprocessing(
        train_src_dir, train_tgt_dir, max_token_seq_len, train_json
    )
    test_set = preprocessing(test_src_dir, test_tgt_dir, max_token_seq_len, test_json)
    voca = AssemblyVoca(text_path=text_voca_path, code_path=code_voca_path)

    params = get_params(voca)
    model = load_model(pretrained_path, params)

    # Select embeddings to replace
    replace_lang_emb(model, params)

    # wrap model with our schema
    model = BinAdapter(model)

    # freeze trained params only for (2)nd and (3)rd scenario
    # comment for training under scenario (1)
    freeze_params(model, only_src=True)

    # Train model using AsmDepictor trainer method in https://github.com/agwaBom/AsmDepictor/blob/main/learn_model_from_scratch.py#L137

    """
    model.load_state_dict(torch.load(
        '/path/to/adapter/state/'), strict=False)
    
    model.module.encoder.src_word_emb = torch.load(
        '/path/to/src_embedding/')    
    """
