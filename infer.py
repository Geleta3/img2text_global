import torch
import random
import json
from config import *
from PIL import Image
import torch.nn.functional as F
from data_loader import get_loader
from model import CapEncoder, CaptionDecoder
from train import transform
from _utils_ import trg_mask
from beam import beam_search
from _utils_ import plot_grounding
from evaluate import blue_score, evaluate

torch.manual_seed(2809)


def test(encoder,
         decoder,
         conf,
         image,
         end=1,
         use_beam=False):

    if use_beam:
        return beam_search(image=image,
                           encoder=encoder,
                           decoder=decoder,
                           conf=conf,
                           end=end)

    else:
        feat = encoder(image)
        encoded, weights = decoder.encoder_att(feat=feat)
        encoded = decoder.encoder(encoded)
        predicted = torch.zeros(1, conf.max_dec_seq).long()
        for i in range(1, conf.max_dec_seq):
            trg_mask_ = trg_mask(predicted[0, :i].unsqueeze(0))
            out = decoder.decoder(memory=encoded, trg=predicted[0, :i].unsqueeze(0), trg_mask=trg_mask_)
            out = F.softmax(out, dim=-1)
            # print("Top-3", out.data.topk(3))
            out = torch.argmax(out, dim=-1)
            if out[..., -1] == end:
                return predicted[0, :i + 1].tolist(), weights
            predicted[0, i] = out[0, -1]

        return predicted[0].tolist(), weights


def clean_sentence(output, type="folder", data_loader=None):
    """
    params:
            output: the index of predicted words.
            type: You wanna generate caption for an image or folder. ["image" or "folder"]
            data_loader: test image folder data loader
    returns concatenated, punctuation filtered sentences
    """
    if type == "folder":
        vocab_ = data_loader.dataset.vocab.idx2word
    elif type == "image":
        with open("vocab.pkl", "rb") as f:
            vocab_ = pickle.load(f)
        vocab_ = vocab_.idx2word
    words = [vocab_.get(idx) for idx in output]
    words = [word for word in words if word not in (',', '.', '<end>', '<start>')]
    sentence = " ".join(words)

    return sentence


def generate_caption(encoder,
                     decoder,
                     conf,
                     type="image",
                     end=1,
                     image_path="",
                     name=None,
                     specific_folder=None,
                     save=False,
                     use_beam=False):

    if type == "image":
        assert image_path != "", " If you want to test a single image path should be provided."
        image = Image.open(image_path).convert('RGB')
        transorm_ = transform(train=False)
        image = transorm_(image).unsqueeze(0)

        output, weights = test(encoder=encoder,
                               decoder=decoder,
                               conf=conf,
                               image=image,
                               end=end,
                               use_beam=use_beam)

        sentence = clean_sentence(output, type)
        print(sentence) if name is None else print(name, "\t", sentence)    # Print the file name and the sentence.
        return sentence, weights
    elif type == "folder":
        save_caption_path = os.path.join(conf.result, conf.save_caption)
        save_caption_file = open(save_caption_path, "w")

        test_data_loader = get_loader(transform=transform(train=False),
                                      batch_size=1,
                                      mode='test',
                                      vocab_file='vocab.pkl',
                                      specific_folder=specific_folder)

        assert test_data_loader is not None, "Data loader can't be None to detect from folder."

        while next(iter(test_data_loader)) is not None:
            orig_image, image, name = next(iter(test_data_loader))

            output, weights = test(encoder=encoder,
                                   decoder=decoder,
                                   conf=conf,
                                   image=image,
                                   end=end,
                                   use_beam=use_beam)

            sentence = clean_sentence(output, type, test_data_loader)
            if save:
                save_caption_file.write(str(name) + "\t" + sentence + "\n")
                save_caption_file.flush()
            print(name, "\t", sentence)
        save_caption_file.close()

        return sentence, weights
    else:
        raise "Choose either image or folder"


def choose_one_img_from_folder(folder,):
    """
    params: folder: Give folder where images are found.
    returns: choose one image randomly and return the path of the image.
    """
    imgs = os.listdir(folder)
    choose_one = random.choice(imgs)
    return os.path.join(folder, choose_one), choose_one

# def get_captions(ann_folder, ann_file, img_file):
#     coco = COCO(ann_folder)
#     annotations = coco.load_caption(ann_file)
#     return annotations[img_file]['caption']


def get_captions(filepath, img_file):
    with open(filepath, 'r') as f:
        print("loading gt captions...")
        annotation = json.load(f)
    return annotation[img_file]["caption"]


if __name__ == '__main__':
    # image_path = "C.../test-burger.jpg"
    # test_folder = "C:.../test/"
    test_folder = "D:/an NPU Stuffs/Research/Scene graph generation/Dataset/coco/val2017/"

    ann_file = "cap_bbox_val2017.json"
    test_coco_val = True
    image_path, name = choose_one_img_from_folder(test_folder)

    with open("vocab.pkl", 'rb') as f:
        vocab = pickle.load(f)
        vocab_size = len(vocab)

    conf = Config(vocab_size=vocab_size)
    encoder_ = CapEncoder()
    decoder_ = CaptionDecoder(conf)

    encoder_.load_state_dict(torch.load(os.path.join(conf.result_path, conf.encoder_file), map_location="cpu"))
    decoder_.load_state_dict(torch.load(os.path.join(conf.result_path, conf.decoder_file), map_location="cpu"))
    encoder_.eval()
    decoder_.eval()

    if not conf.eval_mode:
        # For testing. Not evaluating.
        sentence, weights = generate_caption(conf=conf,
                                             encoder=encoder_,
                                             decoder=decoder_,
                                             type="image",
                                             end=vocab.word2idx.get('<end>'),
                                             image_path=image_path,
                                             name=name,
                                             specific_folder=test_folder,
                                             save=True,
                                             use_beam=False)
        if test_coco_val:
            # If the images you want is from coco dataset.
            ground_truth_captions = get_captions(filepath=ann_file,
                                                 img_file=name)
            print("\nGT *********************")
            for gt in ground_truth_captions:
                print("*\t", gt)
            print("************************")

        # Plotting If an Image is given.
        plot_grounding(image=image_path,
                       weights=weights,
                       caption_generated=sentence,
                       resized_to=(320, 320),
                       k=1)

    # If We want evaluating.
    if conf.eval_mode:
        blue_score(conf=conf,
                   encoder=encoder_,
                   decoder=decoder_,
                   )
