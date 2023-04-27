import string
import matplotlib.pyplot as plt
import numpy as np


def punctuation_free(reference):
    """Function takes a caption and outputs punctuation free and lower cased caption"""
    text = reference.split()
    x = [''.join(c.lower() for c in s if c not in string.punctuation) for s in text]
    return x


def get_batch_caps(caps_all, batch_size):
    """Function takes sampled captions for images and
    returns punctuation-free preprocessed caps in batches"""
    batch_caps_all = []
    for batch_idx in range(batch_size):
        batch_caps = [i for i in map(lambda t: (punctuation_free(t[batch_idx])), caps_all)]
        batch_caps_all.append(batch_caps)
    return batch_caps_all


def get_hypothesis(terms_idx, data_loader):
    """Function outputs word tokens from output indices (terms_idx)
    """
    hypothesis_list = []
    vocab = data_loader.dataset.vocab.idx2word
    # print("Term: ", terms_idx)
    for i in range(terms_idx.size(0)):
        words = [vocab.get(idx.item()) for idx in terms_idx[i]]
        words = [word for word in words if word not in (',', '.', '<end>')]
        hypothesis_list.append(words)
    return hypothesis_list


def visualize_attention(orig_image, words, atten_weights):
    """Plots attention in the image sample.
    
    Arguments:
    ----------
    - orig_image - image of original size
    - words - list of tokens
    - atten_weights - list of attention weights at each time step 
    """
    fig = plt.figure(figsize=(14,12)) 
    len_tokens = len(words)
    
    for i in range(len(words)):
        atten_current = atten_weights[i].detach().numpy()
        atten_current = atten_current.reshape(7,7)       
        ax = fig.add_subplot(len_tokens//2, len_tokens//2, i+1)
        ax.set_title(words[i])
        img = ax.imshow(np.squeeze(orig_image))
        ax.imshow(atten_current, cmap='gray', alpha=0.8, extent=img.get_extent(), interpolation = 'bicubic')
    plt.tight_layout()
    plt.show()


class GetStats:
    """Class for getting statistics from text training/validation files"""
    def __init__(self, log_train_path, log_valid_path, bleu_path):
        self.log_train_path = log_train_path
        self.log_valid_path = log_valid_path
        self.bleu_path = bleu_path
        assert log_train_path == 'training_log.txt', "file must contain training logs named 'training_log.txt'"
        assert log_valid_path == 'validation_log.txt', "file must contain validation logs named 'validation_log.txt'"
        assert bleu_path == 'bleu.txt', "file must contain bleu scores named 'bleu.txt'"

        train_file = open(log_train_path, "r")
        self.train_logs = train_file.readlines()
        train_file.close()

        valid_file = open(log_valid_path, "r")
        self.valid_logs = valid_file.readlines()
        valid_file.close()

        bleu_file = open("bleu.txt", "r")
        self.bleu_score = bleu_file.readlines()
        bleu_file.close()

    def get_train_log(self):
        """Returns training log from training_log.txt file"""
        losses = []
        perplex =[]
        for line in self.train_logs:
            loss = re.search('Loss train: (.*), Perplexity train:', line).group(1)
            losses.append(loss)
            perp = re.search('Perplexity train: (.*)', line).group(1)
            perplex.append(perp)
        return losses, perplex


    def get_valid_log(self):
        """Returns validation log from validation_log.txt file"""
        losses = []
        perplex =[]
        for line in self.valid_logs:
            loss = re.search('Loss valid: (.*), Perplexity valid:', line).group(1)
            losses.append(loss)
            perp = re.search('Perplexity valid: (.*)', line).group(1)
            perplex.append(perp)
        return losses, perplex

    def get_bleu(self):
        """Returns BLEU scores from the text file"""
        bleu_1_scores=[]
        bleu_2_scores=[]
        bleu_3_scores=[]
        bleu_4_scores=[]

        for line in self.bleu_score:
            bleu_1 = re.search('BLEU-1: (.*), BLEU-2:', line).group(1)
            bleu_1_scores.append(bleu_1)

            bleu_2 = re.search('BLEU-2: (.*), BLEU-3:', line).group(1)
            bleu_2_scores.append(bleu_2)

            bleu_3 = re.search('BLEU-3: (.*), BLEU-4:', line).group(1)
            bleu_3_scores.append(bleu_3)

            bleu_4 = re.search('BLEU-4: (.*)', line).group(1)
            bleu_4_scores.append(bleu_4)
        return bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores