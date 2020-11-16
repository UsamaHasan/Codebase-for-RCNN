import os , sys
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

#for unit testing remove after first build.
#sys.path.append('/home/ncai/RoadSurfaceAnalysis/src')
from detection.models.detectors.base import BaseDetector
from detection.utils.config import *
from detection.models.utils.model_visualization import viz
from detection.utils.config import *
import warnings

class Yolov3(BaseDetector):
    """
    Implementation of YOLO model for object detection also known as darknet.
    """
    def __init__(self, cfg_path = '',weights_path = ''):
        """ 
        The following function will initializa a torch model, you can either initialize the model and 
        set the weights here. Or can the respective functions later.
        Args:(Optional)
            weights_path(str): path to weight file to initialize weights.(default='')
            cfg_path(str): path to cfg file.(default=[])
        """
        if(cfg_path!=''):
            super(Yolov3,self).__init__(cfg_path)
        else:
            super(Yolov3,self).__init__(YOLO_V3_CFG_PATH)
        if(weights_path!=''):
            self.load_weights(weights_path)
            
        
    def forward(self,x:Tensor,targets=None) -> Tensor:
        """
        Recevices Image as Tensor and return bbox 
        Args:
            input(Tensor): Image as input tensor.
        Returns:
        """

        input_dim = x.size(2)
        loss = 0.0
        yolo_output = []
        layers_output = []
        dict_ = {}
        for module,module_dict in zip(self.modules_list,self.module_dicts):
            if module_dict['type'] in ['convolutional','upsample','maxpool']:
                x = module(x)
            elif module_dict['type'] == 'shortcut':
                #Add layers output
                idx = int(module_dict['from']) #Add layer 'from'
                x = layers_output[-1] + layers_output[idx]  
            elif module_dict['type'] == 'route':
                #Concat layer
                x = torch.cat([layers_output[int(layer)] for layer in module_dict['layers'].split(',')]\
                    ,1)
            elif module_dict['type'] == 'yolo':
                #yolo Layer.
                x , layer_loss = module[0](x,targets,input_dim)
                loss += layer_loss
                yolo_output.append(x)
            layers_output.append(x)
        yolo_output = (torch.cat(yolo_output,1)).to('cpu')
        return yolo_output if targets is None else (yolo_output,loss)
        #should raise not Implemented error
    
    def load_weights(self,weight_file):
        """
        YOLO Legacy function to load weights from .weights file.
        Args:
            weight_file(str) : path to weight file
        """
        warnings.warn('load_weights is going to be shifted to Legacy yolo version')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        idx = 0

        """Parses and loads the weights stored in 'weights_path'"""
        if isinstance(weight_file,str):
            # Open the weights file
            with open(weight_file, "rb") as f:
                header = np.fromfile(f, dtype=np.int32, count=5)  # First five are header values
                weights = np.fromfile(f, dtype=np.float32)  # The rest are weights
            
        
            ptr = 0
            for i, (module_def, module) in enumerate(zip(self.module_dicts,self.modules_list)):
        
                if module_def["type"] == "convolutional":
                    conv_layer = module[0]
                    if module_def["batch_normalize"]:
                        # Load BN bias, weights, running mean and running variance
                        bn_layer = module[1]
                        num_biases = bn_layer.bias.numel()  # Number of biases
                        # Bias
                        bn_bias = torch.from_numpy(weights[ptr : ptr + num_biases]).view_as(bn_layer.bias)
                        bn_layer.bias.data.copy_(bn_bias)
                        ptr += num_biases
                        # Weight
                        bn_weights = torch.from_numpy(weights[ptr : ptr + num_biases]).view_as(bn_layer.weight)
                        bn_layer.weight.data.copy_(bn_weights)
                        ptr += num_biases
                        # Running Mean
                        bn_runningMean = torch.from_numpy(weights[ptr : ptr + num_biases]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_runningMean)
                        ptr += num_biases
                        # Running Variance
                        bn_runningVariance = torch.from_numpy(weights[ptr : ptr + num_biases]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_runningVariance)
                        ptr += num_biases
                    else:
                        # Load conv. bias
                        num_biases = conv_layer.bias.numel()
                        conv_b = torch.from_numpy(weights[ptr : ptr + num_biases]).view_as(conv_layer.bias)
                        conv_layer.bias.data.copy_(conv_b)
                        ptr += num_biases
                    # Load conv. weights
                    num_weights = conv_layer.weight.numel()
                    conv_weights = torch.from_numpy(weights[ptr : ptr + num_weights]).view_as(conv_layer.weight)
                    conv_layer.weight.data.copy_(conv_weights)
                    ptr += num_weights
        else:                   
            raise TypeError(f'Should be str object')
     
    def forward_train(self):
        """
        """
        raise NotImplementedError
    def forward_test(self):
        """
        """
        raise NotImplementedError

if __name__ == '__main__':
    yolov3 = Yolov3('/home/ncai01/Codebase-of-RCNN/cfg/yolov3.cfg')
    inp = torch.rand(size=(1,3,416,416))
    output = yolov3(inp)
    