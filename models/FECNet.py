from models.inception_resnet_v1 import InceptionResnetV1
from models.densenet import DenseNet
import torch
import torch.nn as nn
import requests
import os
from requests.adapters import HTTPAdapter


class FECNet(nn.Module):
    """FECNet model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the Google facial expression comparison
    dataset (https://ai.google/tools/datasets/google-facial-expression/). Pretrained state_dicts are
    automatically downloaded on model instantiation if requested and cached in the torch cache.
    Subsequent instantiations use the cache rather than redownloading.

    Keyword Arguments:
        pretrained {str} -- load pretraining weights
    """
    def __init__(self, pretrained=False):
        super(FECNet, self).__init__()
        growth_rate = 64
        depth = 100
        block_config = [5]
        efficient = True
        self.Inc = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
        for param in self.Inc.parameters():
            param.requires_grad = False
        self.dense = DenseNet(growth_rate=growth_rate,
                        block_config=block_config,
                        num_classes=16,
                        small_inputs=True,
                        efficient=efficient,
                        num_init_features=512).cuda()

        if (pretrained):
            load_weights(self)

    def forward(self, x):
        feat = self.Inc(x)[1]
        out = self.dense(feat)
        return out

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def load_weights(mdl):
    """Download pretrained state_dict and load into model.

        Arguments:
        mdl {torch.nn.Module} -- Pytorch model."""

    path = 'https://drive.google.com/uc?export=download&id=1iTG-aqh88HBWTWRNN_IAHEoS8J-ns0jx'


    model_dir = os.path.join(os.getcwd(), './pretrained')
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, 'FECNet_AS20112019p.pt')
    if not os.path.exists(cached_file):
        print('Downloading weights')
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()
        id = '1iTG-aqh88HBWTWRNN_IAHEoS8J-ns0jx'

        response = session.get(URL, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)

        save_response_content(response, cached_file)


    mdl.load_state_dict(torch.load(cached_file))