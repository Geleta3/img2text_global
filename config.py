import os
import pickle
is_window = True


class Config:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.d_model = 256
        self.heads = 16
        self.feed_dim = 2048
        self.dropout = 0.1
        self.max_enc_seq = 64
        self.max_dec_seq = 80
        self.dec_layer = 6
        self.enc_layer = 4
        self.enc_input = 10
        self.conv_channel = 2048

        self.enc_dmodel = 128
        self.dec_dmodel = 512

        self.lr = 0.0001    # Not used.
        self.result = "result-1"

        self.result_path = self._result()
        if not is_window:
            self.image_path = "/home/.../coco/"

        else:
            self.image_path = ".../Dataset/coco/"

        self.training_txt = os.path.join(self.result_path, "training_log.txt")
        self.validation_txt = os.path.join(self.result_path, "validation_log.txt")
        self.blue = os.path.join(self.result_path, "blue.txt")

        # training
        self.batch_size = 128
        self.epochs = 100
        self.print_every = 200
        self.save_every = 1
        self.vocab_threshold = 5
        self.vocab_from_file = False
        self.unkown = 1
        self.warmup = 4000

        # testing
        self.k = 3
        self.eval_mode = False

        # saved
        self.encoder_file = 'encoder-0.pkl'
        self.decoder_file = 'decoder-8.pkl'
        self.save_caption = 'generated_caption.txt'

        self._save_parameters("parameters.pkl")

    def _result(self):
        if not is_window:
            path = "..."
        else:
            path = "..."

        path = os.path.join(path, self.result)
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def _save_parameters(self, path):
        with open(os.path.join(self.result_path, path), 'wb') as f:
            pickle.dump(self, f)


# Path to the right path:
