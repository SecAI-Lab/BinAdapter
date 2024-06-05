import sys
from adapter.model import BinAdapter
from utils import *
from config import *
from voca import AssemblyVoca
from train import trainIters


if __name__ == "__main__":
    pretrained_path = sys.argv[-1]

    train_set = preprocessing(
        train_src_dir, train_tgt_dir, max_token_seq_len, train_json
    )
    test_set = preprocessing(test_src_dir, test_tgt_dir, max_token_seq_len, test_json)
    voca = AssemblyVoca(text_path=text_voca_path, code_path=code_voca_path)

    fields = get_fields()
    train_data, valid_data, test_data = get_train_test_data()

    params = get_params(voca)
    model = load_model(pretrained_path, params)

    # Select embeddings to replace
    replace_lang_emb(model, params)

    # wrap model with our schema
    model = BinAdapter(model)

    # freeze trained params
    freeze_params(model, only_src=True)
    print(model)
    exit(0)
    optimizer, train_iterator, valid_iterator = get_optimizer_and_iters(
        model, train_data, valid_data
    )

    trainIters(
        model,
        n_epoch,
        train_iterator,
        valid_iterator,
        optimizer,
        smoothing,
        only_src=True,
    )

    """
    model.load_state_dict(torch.load(
        '/path/to/adapter/state/'), strict=False)
    
    model.module.encoder.src_word_emb = torch.load(
        '/path/to/src_embedding/')    
    """
