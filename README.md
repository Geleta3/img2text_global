# img2text_global
Image Captioning - Global CNN attention + Transformer

Image captioning is describing image with text or sentence. Following explosion of deep learning, image captioning has made tremendous improvement. 
[Karpath](https://arxiv.org/abs/1412.2306) uses global CNN feature as image representation and use RNN as text generation. Following that many researchs
have been carried out [[1]](https://arxiv.org/abs/1411.4555) [[2]](https://arxiv.org/abs/1502.03044) and so on.

[Peter Anderson et al.](https://arxiv.org/abs/1707.07998) introduced using the output of image detection or region proposed from region proposal network. 
Currently this approach of image representation is the state of art result. 

Following [transformer](https://arxiv.org/abs/1706.03762) traditional way of language processing methods like RNN, LSTM has been replaced by this powerful 
algortithm.  It enables to train on large scale dataset and effective parallel computing. [BERT](https://arxiv.org/abs/1810.04805) like models like 
[OSCAR](https://arxiv.org/abs/2004.06165), [LEMON](https://arxiv.org/abs/2111.12233) ... have been proposed to train images on a very large scale 
dataset. 

This repository combines the following algorithms for image captioning training and testing: 
* Golbal CNN feature - Resnet and
* Transformer

The architecture looks like this:

[Model](model-5.png)

To download the dataset visit [here](https://cocodataset.org/#download)

For training run  ``` train.py```

You can edit the configuration in ```config.json```
