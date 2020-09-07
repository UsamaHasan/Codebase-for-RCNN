import torch.nn as nn

def activation_layer(func:str):
    """
    returns activation layer object.
    Args:
        param[in]:func -> str name of activation layer from cfg file.
        param[out]:nn.ReLU or nn.LeakyReLU.
    Raises:
        Exception 
    """
    try:
        if(func=='leaky'):
            return nn.LeakyReLU(negative_slope=0.01)
        elif(func=='linear'):
            return nn.ReLU(inplace=True)
    except Exception:
        raise Exception('Error while initializing Activation Layer')

def conv2d_layer(input_channels:int,dict_:dict) -> nn.Conv2d:
    """
    retruns conv2d layer object.
    Args:
        param[in]:input_channels -> int number of input channels for conv2d.
        param[in]:dict_ -> dict containing parameters of convolutional layer.
        param[out]:conv -> nn.Conv2d convolution 2d layer.
    Raises:
        Key Exception
    """
    try:
        no_filter = int(dict_['filters'])
        kernel_size = int(dict_['size'])         
        stride = int(dict_['stride'])
        pad = (kernel_size - 1) // 2
    except KeyError:
        raise KeyError(f"Error Key not found{dict_['type']} while initializing conv2d layer")
    conv = nn.Conv2d(input_channels,no_filter,kernel_size,stride,pad)
    return conv 

def linear_layer(input_channels,dict_:dict) -> nn.Linear:
    """
    returns linear(MLP) layer object.
    Args:
        param[in]:dict_ -> dict containing parameter of linear layer.
        param[out]:nn.Linear
    Raises:
        Exception
    """
    try:
        return nn.Linear(input_channels,dict['filters'])
    except Exception:
        raise Exception(f'Error while initializing nn.Linear Layer:')

def batchNorm2d_layer(no_filter:int) -> nn.BatchNorm2d:
    """
    returns batchnorm2d layer object.
    Args:
        param[in]:no_filter -> int number of input filter of batchnorm2d.
        param[out]:nn.BatchNorm2d
    Raises:
        Exception 
    """
    try:
        return nn.BatchNorm2d(no_filter)
    except Exception:
        raise Exception('Error while initializing nn.BatchNorm2d Layer:')

def maxpool2d_layer(kernel_size:int) -> nn.MaxPool2d:
    """
    return a nn.MaxPool2d layer.
    Args:
        param[in]:kernel_size -> int kernel_size 
    Raises:
        Exception
    """
    try:
        return nn.MaxPool2d(kernel_size)
    except Exception:
        raise Exception('Error while Initializing nn.MaxPool2d Layer:')

def upsample_layer(no_stride:int) -> nn.Upsample:
    """
    return nn.Upsample as upsampling layer where mode = "nearest" (default setting for yolo).
    Args:
        param[in]:no_stride -> int stride by which upsampling has to be done.
    Raises:
        Exception
    """
    try:
        return nn.Upsample(scale_factor=no_stride,mode="nearest")
    except Exception:
        raise Exception(f'Error caused while initializing nn.Sampling layer')

def route_layer(layers:int , output_filter:list) -> int:
    """
    Implementation of route layer from yolo.
    The layer only update the filter param and will be added inside the nn.ModuleList as Empty Layer
    which is described as a placeholder for these layer.
    Note: The aren't layers of the model.
    Args:
        param[in]: layers -> list
        param[in]:output_filters ->  list contains outputs of previous layers 
        param[out]:filters -> int updated filter of previous layer.
    """
    _layers_ = [int(x) for x in layers.split(",")]
    _filters = sum(output_filter[1:][i] for i in _layers_)
    return _filters
def shortcut_layer(from_:int,output_filter:list) -> list:
    """
    Implementation of shortcut layer from yolo.
    The layer only update the filter param and will be added inside the nn.ModuleList as Empty Layer
    which is described as a placeholder for these layer.
    Note: The aren't layers of the model.
    Args:
        param[in]: from_ -> int a parameter of shortcut layer. 
        param[in]: output_filter -> list contains outputs of previous layers.
        param[out]:filter -> int no_filter
    Raises:
        Exception 
    """
    try:
        return output_filter[1:][from_]
    except Exception:
        raise Exception('Error during shortcut_layer.')

class EmptyLayer(nn.Module):
    """Placeholder for 'route' and 'shortcut' layers"""
    
    def __init__(self):
        super(EmptyLayer, self).__init__()

