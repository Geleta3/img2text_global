import torch
from torch import Tensor
import torch.nn.functional as F
from _utils_ import trg_mask


def first_prediction(image, encoder, decoder, conf, start=0):
    encoder.eval()
    decoder.eval()
    start_token = start
    feat = encoder(image)
    encoded, weights = decoder.encoder_att(feat=feat)

    outputs = Tensor([start_token]).long()

    trg_mask_ = trg_mask(outputs.unsqueeze(0))
    out = decoder.transformer(memory=encoded, trg=outputs.unsqueeze(0), trg_mask=trg_mask_)
    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(conf.k)
    log_scores = torch.log(probs)
    # log_scores = probs
    # print("log: ", log_scores)
    outputs = torch.zeros(conf.k, conf.max_dec_seq).long()
    outputs[:, 0] = start_token
    outputs[:, 1] = ix[0]
    encoded_outputs = torch.zeros(conf.k, encoded.size(-2), encoded.size(-1))
    encoded_outputs[:, :] = encoded[0]
    
    return outputs, encoded_outputs, log_scores, weights


def k_best_outputs(outputs, out, log_scores, i, k):
    probs, ix = out[:, -1].data.topk(k)
    # print("Top-K", out[:, :].data.topk(10))
    # E.g., out -> [[0.9, 0.8, 0.7], [0.7, 0.91, 0.89], [0.4, 0.98, 0.77]]
    log_probs = torch.log(probs) + log_scores
    # log_probs = probs + log_scores
    k_probs, k_ix = log_probs.view(-1).topk(k)
    # print("log prob: ", log_probs)
    row = k_ix // k
    col = k_ix % k
    # print("row, col: ", row, col)
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = ix[row, col]

    log_scores = k_probs.unsqueeze(0)

    return outputs, log_scores


def beam_search(image, encoder, decoder, conf, end=1):
    outputs, en_outputs, log_scores, weights = first_prediction(image,
                                                       encoder,
                                                       decoder,
                                                       conf)
    end_token = end
    ind = None
    # print("Out::", outputs.size(), en_outputs.size(), log_scores.size())
    for i in range(2, conf.max_dec_seq):
        trg_mask_ = trg_mask(torch.LongTensor(i).unsqueeze(0))
        out = decoder.transformer(memory=en_outputs,
                                  trg=outputs[:, :i],
                                  trg_mask=trg_mask_)
        out = F.softmax(out, dim=-1)
        outputs, log_scores = k_best_outputs(outputs,
                                             out,
                                             log_scores,
                                             i,
                                             conf.k)
        none_end = torch.nonzero(outputs == end_token)
        sentence_lengths = torch.zeros(len(outputs)).long()

        for vec in none_end:
            i = vec[0]
            if sentence_lengths[i] == 0:
                sentence_lengths[i] = vec[1]

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])
        if num_finished_sentences == conf.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            # print("div: ", log_scores)
            _, ind = torch.max(log_scores*div, 1)
            # print("Ind: ", ind)
            ind = ind.data[0]
            # ind = torch.argmax(log_scores, dim=-1).data[0]
            break

    if ind is None:
        length = torch.nonzero(outputs[0] == end_token)[0]
        sentence = [tok.item() for tok in outputs[0][1:length]]
        return sentence, weights
    else:
        length = torch.nonzero(outputs[ind] == end_token)[0]
        sentence = [tok.item() for tok in outputs[ind][1:length]]
        return sentence, weights













