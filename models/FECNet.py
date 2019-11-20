from inception_resnet_v1 import InceptionResnetV1
from densenet import DenseNet
import torch
import torch.nn as nn
import requests


class FECNet(nn.Module):
    """FECNet model with optional loading of pretrained weights.

    Model parameters can be loaded based on pretraining on the Google facial expression comparison
    dataset (https://ai.google/tools/datasets/google-facial-expression/). Pretrained state_dicts are
    automatically downloaded on model instantiation if requested and cached in the torch cache.
    Subsequent instantiations use the cache rather than redownloading.

    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None):
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

        if pretrained is not None:
            load_weights(self)

    def forward(self, x):
        feat = self.Inc(x)[1]
        out = self.dense(feat)
        return out

    def load_weights(mdl):
        """Download pretrained state_dict and load into model.

            Arguments:
            mdl {torch.nn.Module} -- Pytorch model."""

        path = 'https://drive.google.com/uc?export=download&id=1cWLH_hPns8kSfMz9kKl9PsG5aNV2VSMn'

        model_dir = os.path.join(os.getcwd(), '/pretrained')
        os.makedirs(model_dir, exist_ok=True)

        state_dict = {}
        cached_file = os.path.join(model_dir, '{}_{}.pt'.format(name, path[-10:]))
        if not os.path.exists(cached_file):
            print('Downloading weights')
            s = requests.Session()
            s.mount('https://', HTTPAdapter(max_retries=10))
            r = s.get(path, allow_redirects=True)
            with open(cached_file, 'wb') as f:
                f.write(r.content)
        state_dict.update(torch.load(cached_file))

        mdl.load_state_dict(state_dict)