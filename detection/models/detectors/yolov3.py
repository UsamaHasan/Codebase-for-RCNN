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
        Args:
            weight_file(str) : path to weight file
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        idx = 0
        if isinstance(weight_file,str):
            with open(weight_file,'rb') as w:
                #header = np.fromfile(w,count=5)
                weights = np.fromfile(w,dtype=np.float32)

            for model_dict , module  in zip(self.module_dicts,self.modules_list):
                if model_dict['type'] in CONVOLUTIONAL:
                    conv_layer = module[0]
                    
                    if model_dict['batch_normalize']:
                        bn_layer = module[1]
                        num_bias = bn_layer.bias.numel()
                        num_weights = bn_layer.weight.numel()
                        #biases.
                        bn_biases = torch.from_numpy(weights[idx:idx+num_bias]).view_as(bn_layer.bias)
                        bn_layer.bias.data.copy_(bn_biases)
                        idx+=num_bias
                        #weights
                        bn_weights = torch.from_numpy(weights[idx:idx+num_weights]).view_as(bn_layer.weight)
                        bn_layer.weight.data.copy_(bn_weights)
                        idx+=num_weights
                        #Running Mean
                        bn_mean = torch.from_numpy(weights[idx:idx+num_bias]).view_as(bn_layer.running_mean)
                        bn_layer.running_mean.data.copy_(bn_mean)
                        idx+=num_bias
                        #Running_Variance
                        bn_variance = torch.from_numpy(weights[idx:idx+num_bias]).view_as(bn_layer.running_var)
                        bn_layer.running_var.data.copy_(bn_variance)
                        idx+=num_bias
                    else:
                        #Conv_layer biases
                        num_bias = conv_layer.bias.numel()
                        conv_bias = torch.from_numpy(weights[idx:idx+num_bias]).view_as(conv_layer.bias)
                        conv_layer.bias.data.copy_(conv_bias)
                        idx+=num_bias
                    
                    num_weights = conv_layer.weight.numel()
                    conv_weights = torch.from_numpy(weights[idx:idx+num_weights]).view_as(conv_layer.weight)
                    
                    conv_layer.weight.data.copy_(conv_weights)
                    idx+=num_weights
        else:                   
            raise TypeError(f'Should be str object')

    def _init_model(self):
        """
        private function to initialize model after cfg has initialized.
        """
        pass
    def forward_train(self):
        """
        """
        raise NotImplementedError
    def forward_test(self):
        """
        """
        raise NotImplementedError
