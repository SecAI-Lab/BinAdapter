# path configs
text_voca_path = "./checkpoint/text.pkl"
code_voca_path = "./checkpoint/code.pkl"
train_src_dir = "./dataset/train_source.txt"
test_src_dir = "./dataset/test_source.txt"
train_tgt_dir = "./dataset/train_target.txt"
test_tgt_dir = "./dataset/test_target.txt"
train_json = "./dataset/train.json"
test_json = "./dataset/test.json"

# model configs
d_word_vec = 512
d_model = 512
device = "cuda"
max_token_seq_len = 300
batch_size = 90
n_epoch = 20
batch_size = 90
d_inner_hid = 2048
d_k = 64
d_v = 64
dropout = 0.1
embs_share_weight = True
n_head = 8
n_layers = 3
proj_share_weight = True
scale_emb_or_prj = "emb"

lr_mul = 1.0
n_warmup_steps = 16000
smoothing = True
