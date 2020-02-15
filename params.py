import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


epochs = 10
batch_size = 4

test_params = [
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
            'folder': 'DenseNet201_CrossEntropy_SGDMomentum',
            'network': 'DenseNet201',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.densenet201(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'GoogleNet_CrossEntropy_SGDMomentum',
            'network': 'GoogleNet',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.googlenet(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': True,
        'name': {
            'folder': 'ResNet152_CrossEntropy_SGDMomentum',
            'network': 'ResNet152',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.resnet152(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': False,
        'name': {
            'folder': 'InceptionV3_CrossEntropy_SGDMomentum',
            'network': 'InceptionV3',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.inception_v3(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': False,
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
        'skip': False,
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
    {
        'skip': False,
        'name': {
            'folder': 'MobileNetV2_CrossEntropy_SGDMomentum',
            'network': 'MobileNetV2',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.mobilenet_v2(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
    {
        'skip': False,
        'name': {
            'folder': 'SqueezeNet1.1_CrossEntropy_SGDMomentum',
            'network': 'SqueezeNet1.1',
            'criterion': 'CrossEntropyLoss',
            'optimizer': 'SGD with momentum',
        },
        'epochs': epochs,
        'batch_size': batch_size,
        'network': lambda: models.squeezenet1_1(),
        'criterion': lambda: nn.CrossEntropyLoss(),
        'optimizer': lambda parameters: optim.SGD(parameters, lr=0.001, momentum=0.9)
    },
]