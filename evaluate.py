import math
import sys
import torch
import numpy as np
import torch.utils.data as data
from _utils_ import device
from utils import get_batch_caps, get_hypothesis
from nltk.translate.bleu_score import corpus_bleu
from _utils_ import trg_mask
from data_loader import get_loader
from _utils_ import transform


def evaluate(conf,
             encoder,
             decoder,
             vdata_loader,
             on_training=False,
             epoch=None,  # Parameter below these are only used in training mode.
             criterion=None,
             num_epochs=None,
             write_file=None,
             bleu_score_file=None,

             ):
    """ Validation function for a single epoch.
    Arguments:
    ----------
    - epoch - number of current epoch
    - encoder - model's Encoder (evaluation)
    - decoder - model's Decoder (evaluation)
    - optimizer - model's optimizer (Adam in our case)
    - criterion - optimized loss function
    - num_epochs - total number of epochs
    - data_loader - specified data loader (for training, validation or test)
    - write_file - file to write the validation logs
    """
    epoch_loss = 0.0
    epoch_perplex = 0.0
    references = []
    hypothesis = []
    total_step_valid = math.ceil(
        len(vdata_loader.dataset.caption_lengths) / vdata_loader.batch_sampler.batch_size)

    # total_step_valid
    for i_step in range(1, total_step_valid + 1):

        # Sample Caption length, create and assign batch sampler.
        vdata_loader = vdata_loader
        indices = vdata_loader.dataset.get_indices()
        new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        vdata_loader.batch_sampler.sampler = new_sampler

        encoder.eval()
        decoder.eval()

        val_images, val_captions, caps_all = next(iter(vdata_loader))
        val_captions_target = val_captions[:, 1:].to(device)
        val_captions = val_captions[:, :-1].to(device)
        val_images = val_images.to(device)
        trg_mask_ = trg_mask(val_captions)
        features = encoder(val_images)
        outputs_val = decoder(caption=val_captions,
                              feat=features,
                              trg_mask=trg_mask_)

        # preprocess captions and add them to the list, form hypothesis
        caps_processed = get_batch_caps(caps_all, batch_size=1)
        references.append(caps_processed[0])
        terms_idx = torch.argmax(outputs_val, dim=-1)
        hyp_list = get_hypothesis(terms_idx, data_loader=vdata_loader)
        hypothesis.append(hyp_list[0])

        if on_training:
            loss_val = criterion(outputs_val.view(-1, conf.vocab_size),
                                 val_captions_target.reshape(-1))
            perplex = np.exp(loss_val.item())
            epoch_loss += loss_val.item()
            epoch_perplex += perplex

            stats = 'Epoch valid: [%d/%d], Step valid: [%d/%d], Loss valid: %.4f, Perplexity valid: %5.4f' % (
                epoch, num_epochs, i_step, total_step_valid, loss_val.item(), perplex)

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            # Print training statistics to file.
            write_file.write(stats + '\n')
            write_file.flush()

            # Print training statistics (on different line).
            # if i_step % conf.print_every == 0:
            #     print('\r' + stats)
        else:
            print("\r", "time-step: ", i_step, f"/{total_step_valid}", end="")
            sys.stdout.flush()

    epoch_loss_avg = epoch_loss / total_step_valid
    epoch_perp_avg = epoch_perplex / total_step_valid

    bleu_1 = corpus_bleu(references, hypothesis, weights=(1.0, 0, 0, 0))
    bleu_2 = corpus_bleu(references, hypothesis, weights=(0.5, 0.5, 0, 0))
    bleu_3 = corpus_bleu(references, hypothesis, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0))
    bleu_4 = corpus_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25))

    print('\r')
    if on_training:
        print('Epoch valid:', epoch)
        epoch_stat = 'Avg. Loss valid: %.4f, Avg. Perplexity valid: %5.4f,     BLEU-1: %.2f, BLEU-2: %.2f, ' \
                     'BLEU-3: %.2f, BLEU-4: %.2f' % (epoch_loss_avg, epoch_perp_avg, bleu_1, bleu_2, bleu_3, bleu_4)
        bleu_score_file.write(epoch_stat + '\n')
        bleu_score_file.flush()
    else:
        print("BLEU Score")
        epoch_stat = 'BLEU-1: %.2f, BLEU-2: %.2f, BLEU-3: %.2f, ' \
                     'BLEU-4: %.2f' % (bleu_1, bleu_2, bleu_3, bleu_4)

    print('\r' + epoch_stat, end="")
    print('\r')

    return epoch_stat


def blue_score(conf,
               encoder,
               decoder,
               ):
    data_loader = get_loader(transform(train=False),
                             mode='valid',
                             batch_size=conf.batch_size,
                             specific_folder=None)

    blue = evaluate(conf,
                    encoder,
                    decoder,
                    vdata_loader=data_loader,)
    return blue
