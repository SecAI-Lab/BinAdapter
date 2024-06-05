from sklearn.metrics import precision_recall_fscore_support
from model.asmdepictor.Translator import Translator
import torch.nn.functional as F
from tqdm import tqdm
import random
import json
import torch
import math

from config import *
from utils import tokenize
from adapter.model import adapter_state_dict


def cal_performance(
    pred, pred_sentence, gold, gold_sentence, trg_pad_idx, smoothing=False, text=None
):
    loss = cal_loss(pred, gold, trg_pad_idx, smoothing=smoothing)

    pred = pred.max(1)[1]
    pred_sentence = pred_sentence.max(-1)[1]
    gold = gold.contiguous().view(-1)
    pad_mask = gold.ne(trg_pad_idx)
    eos_mask = gold.ne(text.vocab.stoi.get("<eos>"))

    new_mask = pad_mask & eos_mask

    y_pred = pred.masked_select(new_mask).to("cpu")
    y_test = gold.masked_select(new_mask).to("cpu")

    f1 = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    jaccard = True
    if jaccard:
        unique_pred = [set(l) for l in pred_sentence.tolist()]
        unique_gold = [set(l) for l in gold_sentence.tolist()]

        n_correct = 0
        n_word = 0
        e_value = 0
        for i in range(0, len(unique_gold)):
            union = unique_gold[i]
            if trg_pad_idx in union:
                union.remove(trg_pad_idx)
                union.remove(2)  # remove special token for eval
                union.remove(3)  # remove special token for eval

            intersection = unique_pred[i].intersection(unique_gold[i])
            if trg_pad_idx in intersection:
                intersection.remove(trg_pad_idx)
                intersection.remove(2)  # remove special token for eval
                intersection.remove(3)  # remove special token for eval

            alpha = 14
            if len(unique_pred[i]) <= len(unique_gold[i]) + alpha:
                e_value = 1
            else:
                e_value = math.exp(
                    1 - (len(unique_pred[i]) / (len(unique_gold[i]) + alpha))
                )

            n_correct += len(intersection) * e_value
            n_word += len(union)

    else:
        n_correct = pred.eq(gold).masked_select(new_mask).sum().item()
        n_word = new_mask.sum().item()

    return loss, n_correct, n_word, f1


def cal_loss(pred, gold, trg_pad_idx, smoothing=False):
    gold = gold.contiguous().view(-1)
    if smoothing:
        eps = 0.1
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        pad_mask = gold.ne(trg_pad_idx)

        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(pad_mask).sum()  # average later
    else:
        loss = F.cross_entropy(pred, gold, ignore_index=trg_pad_idx, reduction="sum")

    return loss


def train(input_tensor, target_tensor, model, model_optimizer, smoothing):
    model_optimizer.zero_grad()
    target_tensor = target_tensor.transpose(0, 1)
    input_tensor = input_tensor.transpose(0, 1)

    gold = target_tensor[:, 1:].contiguous().view(-1)
    target_tensor = target_tensor[:, :-1]
    dec_output = model(input_tensor, target_tensor)
    dec_output_sentence = dec_output
    dec_output = dec_output.view(-1, dec_output.size(2))

    # backward and update parameters
    loss, n_correct, n_word, f1 = cal_performance(
        dec_output,
        dec_output_sentence,
        gold,
        target_tensor,
        trg_pad_idx,
        smoothing=smoothing,
    )

    loss.backward()
    model_optimizer.step_and_update_lr()

    return loss.item(), n_correct, n_word, f1


def validate(input_tensor, target_tensor, model):
    target_tensor = target_tensor.transpose(0, 1)
    input_tensor = input_tensor.transpose(0, 1)

    gold = target_tensor[:, 1:].contiguous().view(-1)
    target_tensor = target_tensor[:, :-1]
    dec_output = model(input_tensor, target_tensor)
    dec_output_sentence = dec_output
    dec_output = dec_output.view(-1, dec_output.size(2))

    loss, n_correct, n_word, f1 = cal_performance(
        dec_output,
        dec_output_sentence,
        gold,
        target_tensor,
        trg_pad_idx,
        smoothing=False,
    )

    return loss.item(), n_correct, n_word, f1


def trainIters(
    model,
    n_epoch,
    train_iterator,
    test_iterator,
    model_optimizer,
    smoothing,
    only_src=False,
):
    for i in range(1, n_epoch + 1):
        total_train_f1 = 0
        total_train_prec = 0
        total_train_rec = 0
        n_word_total = 0
        n_word_correct = 0
        total_loss = 0
        # one epoch
        print("\n", i, " Epoch...")
        print("\nTraining loop")
        model.module.train()
        for batch in tqdm(train_iterator):
            input_tensor = batch.code
            target_tensor = batch.text
            train_loss, n_correct, n_word, f1 = train(
                input_tensor, target_tensor, model, model_optimizer, smoothing
            )
            total_train_prec += f1[0]
            total_train_rec += f1[1]
            total_train_f1 += f1[2]

            n_word_total += n_word
            n_word_correct += n_correct
            total_loss += train_loss

        loss_per_word = total_loss / n_word_total
        accuracy = n_word_correct / n_word_total

        mean_train_prec = total_train_prec / len(train_iterator)
        mean_train_rec = total_train_rec / len(train_iterator)
        mean_train_f1 = total_train_f1 / len(train_iterator)

        print(
            "loss : ",
            loss_per_word,
            "Jaccard* : ",
            accuracy,
            "F1 : ",
            mean_train_f1,
            "Precision : ",
            mean_train_prec,
            "Recall : ",
            mean_train_rec,
        )

        # Validation loop
        print("\nValidation loop")
        model.module.eval()
        total_valid_f1 = 0
        total_valid_prec = 0
        total_valid_rec = 0
        n_word_total = 0
        n_word_correct = 0
        total_loss = 0
        with torch.no_grad():
            for batch in tqdm(test_iterator):
                input_tensor = batch.code
                target_tensor = batch.text
                target_loss, n_correct, n_word, f1 = validate(
                    input_tensor, target_tensor, model
                )
                total_valid_prec += f1[0]
                total_valid_rec += f1[1]
                total_valid_f1 += f1[2]

                n_word_total += n_word
                n_word_correct += n_correct
                total_loss += target_loss

            loss_per_word = total_loss / n_word_total
            accuracy = n_word_correct / n_word_total

            mean_valid_f1 = total_valid_f1 / len(test_iterator)
            mean_valid_rec = total_valid_rec / len(test_iterator)
            mean_valid_prec = total_valid_prec / len(test_iterator)

            print(
                "loss : ",
                loss_per_word,
                "Jaccard* : ",
                accuracy,
                "F1 : ",
                mean_valid_f1,
                "Precision : ",
                mean_valid_prec,
                "Recall : ",
                mean_valid_rec,
            )

            # Random select data in train and check training
            src_rand_train, tgt_rand_train = random_choice_from_train(test_json)
            train_hypothesis = make_a_hypothesis_transformer(model, src_rand_train)
            print("Expected output : ", tgt_rand_train)
            print("Hypothesis output : ", "".join(train_hypothesis))

    print("Saving adapter ....")
    torch.save(adapter_state_dict(model), "adapter_1.param")
    torch.save(model.module.encoder.src_word_emb, "src_emb_1.param")

    if not only_src:
        torch.save(model.module.decoder.trg_word_emb, "trg_emb_1.param")
        torch.save(model.module.trg_word_prj, "trg_word_prj_1.param")


def random_choice_from_train(json_file):
    train_data = list()
    for line in open(json_file, mode="r", encoding="utf-8"):
        train_data.append(json.loads(line))

    train_data = random.choice(train_data)
    source = train_data["Code"].lower()
    target = train_data["Text"].lower()

    return source, target


def sentence_to_tensor(sentence, code=None):
    sentence = tokenize(sentence)
    unk_idx = code.vocab.stoi[code.unk_token]
    pad_idx = code.vocab.stoi[code.pad_token]
    sentence_idx = [code.vocab.stoi.get(word, unk_idx) for word in sentence]
    for i in range(max_token_seq_len - len(sentence_idx)):
        sentence_idx.append(code.vocab.stoi.get(i, pad_idx))

    sentence_tensor = torch.tensor(sentence_idx).to(device)
    sentence_tensor = sentence_tensor.unsqueeze(0)
    return sentence_tensor


def make_a_hypothesis_transformer(model, src, text, sos_idx=None, eos_idx=None):
    input_tensor = sentence_to_tensor(src)
    translator = Translator(
        model=model,
        beam_size=5,
        max_seq_len=max_token_seq_len + 3,
        src_pad_idx=src_pad_idx,
        trg_pad_idx=trg_pad_idx,
        trg_bos_idx=sos_idx,
        trg_eos_idx=eos_idx,
    ).to(device)

    output_tensor = translator.translate_sentence(input_tensor)
    try:
        predict_sentence = " ".join(text.vocab.itos[idx] for idx in output_tensor)
    except:
        predict_sentence = " "
    predict_sentence = predict_sentence.replace("<sos>", "").replace("<eos>", "")
    return predict_sentence


def make_hypothesis_reference(model, test_src, test_tgt):
    hypothesis_list = list()
    reference_list = test_tgt
    print("Building hypothesis list...")
    for src, tgt in tqdm(zip(test_src, test_tgt)):
        hypothesis = make_a_hypothesis_transformer(model, src)
        hypothesis_list.append(hypothesis)
    return hypothesis_list, reference_list
