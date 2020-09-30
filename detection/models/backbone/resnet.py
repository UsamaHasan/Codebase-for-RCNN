import torch.nn as nn
import torch
def conv3x3(in_channels,out_channels,stride=1,dilation=1,groups=1):
    """
    Conv3*3 
    """
    return nn.Conv2d(in_channels,out_channels,3,stride,padding=dilation,\
        dilation=dilation ,groups=groups,bias=True)

def conv1x1(in_channels,out_channels,stride):
    """
    Conv1*1
    """
    return nn.Conv2d(in_channels,out_channels,1,stride,bias=True)

class Basicblock(nn.Module):
    """
    Basic Block of resnet
    """
    expansion = 1
    def __init__(self,in_channel,out_channel,stride=1,downsample=None,groups=1,norm_layer=None,\
        ,dilation=1,base_width=64):

        super(BasicBlock,self).__init__()
        if groups!= 1 or base_width!=64:
            raise 'Basic Block doesnot support either groups>1 and basewidth!=64.'
        if dilation>1:
            raise 'Dilation cannot be greater then 1 for Basic Block.'
        if norm is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv3x3(in_channel,in_channel,stride)
        self.bn1 = norm_layer(in_channel)
        self.conv2 = conv3x3(in_channel,out_channel)
        self.bn2 = norm_layer(out_channel)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(x))
        out = self.relu(out)

        if downsample is not None:
            identity = self.downsample(x)
        out+=identity
        return out

class BottleNeck(nn.Module):
    """

    """
    expansion = 4
    def __init__(self,inplanes,planes,stride=1,downsample=None,norm_layer=None,groups=1,\
        dilation = 1,base_width=64 ):
        super(BottleNeck,self).__init__()
        if norm is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * base_width/64.) * groups
        self.conv1 = conv1x1(inplanes,width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width,width,stride)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width ,width * expansion)
        self.bn3  = norm_layer(width*expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
        
    def forward(self,x):
        identity = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out)
        out = self.bn3(self.conv3(out))
        out = self.relu(out)
        if self.downsample is not None:
            identity  = self.downsample(x)
        out += identity
        return out  

class Resnet(nn.Module):
    def __init__(self,block,layers,num_classes=1000,zero_init_residual=False,groups=1,\
        width_per_group=64,norm_layer):
        """
        Args:
            block(BasicBlock/BottleNeck) : Type of residual block to be appended in the network.
            layer(list) : Number of block.
            num_classes(int) : Number of classes to be classified default 1000 as in the original implementation for Imagenet.
            zero_init_residual(bool):
            groups(int) : default 1.
            width_per_group (int) : default 64.
        """
        super(Resnet,self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.norm_layer = norm_layer
        self.dilation = 1
        self.inplanes = 64
        self.base_width=width_per_group
        self.groups = groups
        self.conv1 = nn.Conv2d(3,self.inplanes,7,stride=2,padding=3,bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(True)

    def residual_block(self,block,planes,blocks,stride=1):
        """
        """
        norm_layer = self.norm_layer
        dilation=1
        downsample=None
        if stride!=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes,planes*block.expansion,stride),
                norm_layer(planes*block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes,planes,stride,downsample,groups=self.groups,base_width=self.base_width))
        self.inplanes = planes * block.expansion
        for i in range(0,blocks):
            layers.append(block(self.inplanes,planes,stride,downsample,groups=self.groups,base_width=self.base_width))
    def _resnet(block,planes,blocks,pretrained,**kwargs):
        """"""
        model = Resnet()
        pass

    def resnet18():
        """
        """
        pass
    def resnet34():
        """
        """
        pass
    def resnet50():
        """
        """
        pass
    def resnet101():
        """
        """
        pass
    def resnet152():
        """
        """        
        pass
if __name__ == '__main__':
    
    pass



