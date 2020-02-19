import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from models.sigmoid_alexnet import SigmoidAlexNet
from models.leaky_alexnet import LeakyAlexNet
from models.softplus_alexnet import SoftPlusAlexNet
from torch.optim.rmsprop import RMSprop

epochs = 10
batch_size = 4

test_params = [
    {
        'skip': False,
        'name': {
            'folder': 'AlexNet_CrossEntropy_SGDMomentum_32batch',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGDMomentum',
        },
        'epochs': epochs,
        'batch_size': 32,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': False,
        'name': {
            'folder': 'AlexNet_CrossEntropy_SGDMomentum_128batch',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGDMomentum',
        },
        'epochs': epochs,
        'batch_size': 128,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_Adam',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'Adam',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.Adam(parameters, lr=0.0001)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_RMSProp_wd0',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'RMSProp wd: 0',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: RMSprop(parameters, lr=0.0001, weight_decay=0)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_RMSProp_wd.5',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'RMSProp wd: 0.5',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: RMSprop(parameters, lr=0.0001, weight_decay=.5)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_RMSProp_wd.9',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'RMSProp wd: 0.9',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: RMSprop(parameters, lr=0.0001, weight_decay=.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_SGDMomentum_LeakyReLU',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: LeakyAlexNet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_SGDMomentum_SoftPlus',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: SoftPlusAlexNet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_SGDMomentum_Sigmoid',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: SigmoidAlexNet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'AlexNet_CrossEntropy_SGDMomentum',
            'network': 'AlexNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.alexnet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'VGG19_CrossEntropy_SGDMomentum',
            'network': 'VGG19',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.vgg19(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'InceptionV3_CrossEntropy_SGDMomentum',
            'network': 'InceptionV3',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.inception_v3(),
        'criterion': lambda: nn.BCEWithLogitsLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'ShuffleNetV2X1_CrossEntropy_SGDMomentum',
            'network': 'ShuffleNetV2X1',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.shufflenet_v2_x1_0(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'MnasNet1.3_CrossEntropy_SGDMomentum',
            'network': 'MnasNet1.3',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.mnasnet1_3(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    # {
    #     'skip': False,
    #     'name': {
    #         'folder': 'GoogleNet_CrossEntropy_SGDMomentum',
    #         'network': 'GoogleNet',
    #         'criterion': 'CrossEntropyLoss',
    #         'optimizer': 'SGD with momentum',
    #     },
    #     'epochs': epochs,
    #     'batch_size': batch_size,
    #     'network': lambda: models.googlenet(),
    #     'criterion': lambda: nn.BCEWithLogitsLoss(),
    #     'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    # },
    # {
    #     'skip': False,
    #     'name': {
    #         'folder': 'ResNet152_CrossEntropy_SGDMomentum',
    #         'network': 'ResNet152',
    #         'criterion': 'CrossEntropyLoss',
    #         'optimizer': 'SGD with momentum',
    #     },
    #     'epochs': epochs,
    #     'batch_size': batch_size,
    #     'network': lambda: models.resnet152(),
    #     'criterion': lambda: nn.CrossEntropyLoss(),
    #     'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    # },
]
