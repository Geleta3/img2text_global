import math
import torch
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform(train: bool):
    if train:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            # transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    else:
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225))])
    return transform


def trg_mask(trg: torch.Tensor, unwanted=None):
    bs, seq_len = trg.size()
    if unwanted is not None:
        mask_unwanted = (trg != unwanted).long().unsqueeze(-1)
    else:
        mask_unwanted = trg.unsqueeze(-1).bool()

    ones = torch.ones(1, seq_len, seq_len)
    triu = (torch.triu(ones, diagonal=1) == 0).long().to(device)
    mask = triu & mask_unwanted
    mask[:, 0, 0] = 1  # Start = 0 in the Vocab. Helps to be unmasked.
    return mask


def write_caption(caption):
    """
    Used to write part of the caption on new line if it is long.
    """
    end_of_line_len = 40
    len_ch = len(caption)
    if len_ch > end_of_line_len:
        count = 0
        caption = caption.split(sep=" ")
        processed_cap = []
        for cap in caption:
            count += len(cap)
            if count/end_of_line_len>1:
                processed_cap.append("\n")
                processed_cap.append(cap)
                count = len(cap)
            else:
                processed_cap.append(cap)

        caption = " ".join(cap for cap in processed_cap)
        # print(caption)
        return str(caption)
    else:
        return caption


def plot_grounding(image, caption_generated, weights, resized_to, k=10):
    h, w = resized_to
    img = Image.open(image)
    img = img.convert("RGBA")
    img = img.resize((h, w))
    img = np.array(img)

    blured_image = blur(img, weights, k=k)
    white = np.ones([50, w, 4], dtype='uint8') * 240
    v_stacked_img = np.concatenate((blured_image, white))
    img = Image.fromarray(np.uint8(v_stacked_img))
    im = ImageDraw.Draw(img)
    im.text((15, 330), text=write_caption(caption_generated), fill=(20, 20, 20))
    img.show()


def blur(resized_img, weights, k=10):
    _, dim, seq_len = weights.size()
    max_feats, idx = weights.view(1, 1, -1).data.topk(k)
    wh, ww = math.sqrt(dim), math.sqrt(dim)
    row = torch.div(idx, ww, rounding_mode='trunc')
    col = idx % wh

    # [0, 1, 2...] , [0, 32, 64,...] -> [0-> [0-32], 1-> [32-64]...]
    h, w, c = resized_img.shape
    h_factor, w_factor = h//wh, w//ww

    # Mapping.
    row, col = row[0].numpy()*w_factor, col[0].numpy()*h_factor
    resized_img[:, :, -1] = 200
    blurring = 205 - scaling(np.arange(k), 0, 150).astype(np.uint8)
    row, col = row.astype(np.int), col.astype(np.int)
    i = 0
    for r, c in zip(row[0], col[0]):
        resized_img[c:c+32, r:r+32, -1] += blurring[i]
        i += 1

    return resized_img


def scaling(array, min, max):
    min_array = np.min(array)
    max_array = np.array(array)
    rescaled = (array - min_array)/(max_array-min_array+1e-9) * (max-min) + min
    return rescaled








