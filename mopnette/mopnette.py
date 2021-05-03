import torch
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, model_urls
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch._jit_internal import Optional
from fastai.vision.augment import aug_transforms
from fastai.metrics import accuracy
from fastai.vision.data import ImageDataLoaders
from fastai.vision.all import Normalize, imagenet_stats
from fastai.learner import Learner
from pathlib import Path

def accuracy(inp, targ, axis=-1):
    "Compute accuracy with `targ` when `pred` is bs * n_classes"
    pred,targ = flatten_check(inp.argmax(dim=axis), targ)
    return (pred == targ).float().mean()
def _replace_relu(module):
    reassign = {}
    for name, mod in module.named_children():
        _replace_relu(mod)
        # Checking for explicit type instead of instance
        # as we only want to replace modules of the exact type
        # not inherited classes
        if type(mod) == nn.ReLU or type(mod) == nn.ReLU6:
            reassign[name] = nn.ReLU(inplace=False)

    for key, value in reassign.items():
        module._modules[key] = value


def quantize_model(model, backend):
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)

    return

__all__ = ['QuantizableResNet', 'resnet18', 'resnet50']


class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs)

        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        x = self._forward_impl(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                m.fuse_model()


def _resnet(arch, block, layers, pretrained, progress, quantize, **kwargs):
    model = QuantizableResNet(block, layers, **kwargs)
    _replace_relu(model)
    if quantize:
        # TODO use pretrained as a string to specify the backend
        backend = 'fbgemm'
        quantize_model(model, backend)
    else:
        assert pretrained in [True, False]

    if pretrained:
        if quantize:
            model_url = quant_model_urls[arch + '_' + backend]
        else:
            model_url = model_urls[arch]

        state_dict = load_state_dict_from_url(model_url,
                                              progress=progress)

        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, quantize=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _resnet('resnet18', QuantizableBasicBlock, [2, 2, 2, 2], pretrained, progress,
                   quantize, **kwargs)


def resnet50(pretrained=False, progress=True, quantize=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        quantize (bool): If True, return a quantized version of the model
    """
    return _resnet('resnet50', QuantizableBottleneck, [3, 4, 6, 3], pretrained, progress,
                   quantize, **kwargs)

def print_model_size(model):
    """ Print the size of the model.
    
    Args:
        model: model whose size needs to be determined

    """
    torch.save(model.state_dict(), "temp.p")
    print('Size of the model(MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')
    

from fastai.losses import CrossEntropyLossFlat
# md_ef =  EfficientNet.from_pretrained('efficientnet-b4', num_classes=1)
def train_static_quantized_vision_model(epochs, df, label_col, train_path='', save_path='./', arch='resnet18', metrics=[accuracy], loss=CrossEntropyLossFlat, BS=16):
    dls = ImageDataLoaders.from_df(df=df, path=Path('../input/cassava-leaf-disease-classification/train_images/'), cols='image', valid_pct=0.3, label_col=label_col, batch_tfms=Normalize.from_stats(*imagenet_stats), image_tfms = aug_transforms(size=244), bs=BS, val_bs=16)
    if arch=='resnet18':
        learn = Learner(dls, resnet18(pretrained=True), metrics=metrics)
        
    elif arch == 'resnet50':
        learn = Learner(dls, resnet50(pretrained=True), metrics=metrics)
        
    lr = 2e-3

    learn.fit_one_cycle(epochs, slice(lr/100, lr))

    
    torch.save(learn.model.state_dict(),save_path+'temp.pth')
    if arch=='resnet18':
        learn = resnet18()
    elif arch == 'resnet50':
        learn = resnet50()
    learn.load_state_dict(torch.load(save_path+'temp.pth'))
    os.remove('temp.pth')
    print_model_size(learn)
    backend = "qnnpack"

    learn.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.backends.quantized.engine = backend
    model_static_quantized = torch.quantization.prepare(learn, inplace=False)
    model_static_quantized = torch.quantization.convert(model_static_quantized, inplace=False)
    print_model_size(model_static_quantized)
    model_static_quantized.to('cpu')
    torch.save(model_static_quantized.state_dict(),save_path+'quant.pth')
    return model_static_quantized
    