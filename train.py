import sys
import nltk
import math
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.utils.data as data
from data_loader import get_loader
from _utils_ import transform, trg_mask
from model import CapEncoder, CaptionDecoder
from evaluate import evaluate
from config import *

# nltk.download('punkt')


def learning_rate(steps, conf):
    lr = math.sqrt(1 / conf.d_model) * (min(1 / math.sqrt(steps), steps * (math.pow(conf.warmup, -1.5))))
    return lr


def train(conf,
          encoder,
          decoder,
          train_data_loader,
          vdata_loader=None,
          ):

    for p in decoder.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, betas=(0.9, 0.98), lr=1e-4)

    train_write_file = open(conf.training_txt, "w")
    val_write_file = open(conf.validation_txt, "w")
    blue_score_file = open(conf.blue, "w")

    # total_step
    total_step = math.ceil(len(train_data_loader.dataset.caption_lengths) / train_data_loader.batch_sampler.batch_size)
    step = 0
    for epoch in range(conf.epochs):
        epoch_loss = 0.0
        epoch_perplex = 0.0
        for i_step in range(1, 2): #total_step + 1):
            step += 1
            lr = learning_rate(steps=step,
                               conf=conf)
            optimizer.param_groups[0]["lr"] = lr
            encoder.eval()  # no fine-tuning for Encoder
            decoder.train()

            # Randomly sample a caption length, and sample indices with that length.
            indices = train_data_loader.dataset.get_indices()
            new_sampler = data.sampler.SubsetRandomSampler(indices=indices)
            train_data_loader.batch_sampler.sampler = new_sampler

            images, captions = next(iter(train_data_loader))
            captions_target = captions[:, 1:].to(device)
            captions_train = captions[:, :-1].to(device)
            trg_mask_ = trg_mask(captions_train)
            images = images.to(device)

            decoder.zero_grad()
            encoder.zero_grad()

            features = encoder(images)
            outputs = decoder(caption=captions_train,
                              feat=features,
                              trg_mask=trg_mask_)

            loss = criterion(outputs.view(-1, conf.vocab_size), captions_target.reshape(-1))
            loss.backward()
            optimizer.step()

            perplex = np.exp(loss.item())
            epoch_loss += loss.item()
            epoch_perplex += perplex

            stats = 'Epoch train: [%d/%d], Step train: [%d/%d], Loss train: %.4f, Perplexity train: %5.4f' % (
                epoch, conf.epochs, i_step, total_step, loss.item(), perplex)

            # Print training statistics (on same line).
            print('\r' + stats, end="")
            sys.stdout.flush()

            train_write_file.write(stats + '\n')
            train_write_file.flush()

            # Print training statistics (on different line).
            if i_step % conf.print_every == 0:
                torch.save(decoder.state_dict(), os.path.join(conf.result_path, 'decoder-%d.pkl' % epoch))
                print('\r' + stats)

        epoch_loss_avg = epoch_loss / total_step
        epoch_perp_avg = epoch_perplex / total_step

        print('\r')
        print('Epoch train:', epoch)
        print('\r' + 'Avg. Loss train: %.4f, Avg. Perplexity train: %5.4f' % (epoch_loss_avg, epoch_perp_avg), end="")
        print('\r')

        # Save the weights.
        if epoch % conf.save_every == 0:
            torch.save(decoder.state_dict(), os.path.join(conf.result_path, 'decoder-%d.pkl' % epoch))
            # torch.save(encoder.state_dict(), os.path.join(RESULT_ROOT_PATH, 'encoder-%d.pkl' % epoch))

        # Validation:
        if vdata_loader is not None:
            evaluate(conf=conf,
                     encoder=encoder,
                     decoder=decoder,
                     vdata_loader=vdata_loader,
                     on_training=True,
                     epoch=epoch,  # Parameter below these are only used in training mode.
                     criterion=criterion,
                     num_epochs=conf.epochs,
                     write_file=val_write_file,
                     bleu_score_file=blue_score_file)

    train_write_file.close()
    val_write_file.close()
    blue_score_file.close()


if __name__ == '__main__':
    print("Training...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not is_window:
        vocab_file = "/home/geleta/source codes/Without Scene graph/TIC-4/vocab.pkl"
    else:
        vocab_file = "vocab.pkl"

    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    config = Config(vocab_size=vocab_size)
    encoder = CapEncoder()
    decoder = CaptionDecoder(conf=config)
    encoder.to(device)
    decoder.to(device)

    t_data_loader = get_loader(transform=transform(train=True),
                               mode='train',
                               batch_size=config.batch_size,
                               vocab_threshold=config.vocab_threshold,
                               vocab_from_file=config.vocab_from_file)

    v_data_loader = get_loader(transform=transform(train=False),
                               mode='valid',
                               vocab_threshold=config.vocab_threshold,
                               vocab_from_file=config.vocab_from_file,
                               batch_size=1)
    train(conf=config,
          train_data_loader=t_data_loader,
          encoder=encoder,
          decoder=decoder,
          vdata_loader=v_data_loader
          )
