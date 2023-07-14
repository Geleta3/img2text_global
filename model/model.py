import torch
from torch import nn
from torchvision import models
from transformer import Decoder, Encoder
from torch.nn import functional as F
from _utils_ import device


class CapEncoder(nn.Module):
    """
    Pretrained Image feature Extractor-resnet152.
    """

    def __init__(self, ):
        super(CapEncoder, self).__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

    def forward(self, images):
        # Should be resize to [bs, ch, h, w]
        features = self.resnet(images)
        return features


class AttendFeat(nn.Module):
    def __init__(self, conf):
        super(AttendFeat, self).__init__()
        self.layer1 = nn.Linear(conf.d_model, conf.d_model)
        self.layer2 = nn.Linear(conf.d_model, 1)
        self.conv = nn.Conv2d(conf.conv_channel, conf.d_model, kernel_size=(3, 3))

    def forward(self, feat):
        x = self.conv(feat)
        bs, ch, h, w = x.size()
        img_feat = x.view(bs, ch, h * w).transpose(-1, -2)
        x = torch.tanh(self.layer1(img_feat))
        weights = F.softmax(self.layer2(x), dim=-2)
        att = torch.mul(img_feat, weights)
        return att, weights


class CaptionDecoder(nn.Module):
    def __init__(self, conf, device=device):
        super(CaptionDecoder, self).__init__()

        self.encoder_att = AttendFeat(conf)
        self.decoder = Decoder(d_model=conf.d_model,
                               heads=conf.heads,
                               feed_dim=conf.feed_dim,
                               n_layer=conf.dec_layer,
                               vocab_size=conf.vocab_size,
                               max_seq=conf.max_dec_seq,
                               dropout=conf.dropout,
                               device=device)
        self.encoder = Encoder(d_model=conf.d_model,
                               heads=conf.heads,
                               feed_dim=conf.feed_dim,
                               n_layer=conf.enc_layer,
                               max_seq=conf.max_enc_seq,
                               dropout=conf.dropout,
                               device=device)

    def forward(self, feat, caption, src_mask=None, trg_mask=None):
        x, weights = self.encoder_att(feat=feat)
        x = self.encoder(src=x, src_mask=src_mask)
        x = self.decoder(memory=x, trg=caption, trg_mask=trg_mask)

        return x

# Check.
