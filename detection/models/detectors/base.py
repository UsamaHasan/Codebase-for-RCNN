import torch
import torch.nn as nn
import os , sys
import numpy as np
#for unit testing remove after first build.
#sys.path.append('/home/ncai/RoadSurfaceAnalysis/src')
from detection.models.utils.net_utils import *
from detection.utils.utils import parse_model_config
from detection.utils.config import *
from abc import ABCMeta, abstractmethod
import warnings
class BaseDetector(nn.Module,metaclass=ABCMeta):
    """
    A base class for all the RCNN frameworks. The base class will load all the
    modules from the cfg and return nn.ModuleList, further forward function is divided 
    into forward_test and forward_train for convinence. Forward function will do a forward pass
    to the backbone feature detector. The derived will call forward to extract feature and then
    implement there own defination for further calculation. 
    """
    def __init__(self,cfg):
        super(BaseDetector,self).__init__()
        #modules_list will contain all the Sequential modules of model.
        if(cfg != ''):
            self.modules_list , self.output_filters , self.module_dicts = BaseDetector.model_from_cfg(cfg)
        
    @staticmethod    
    def model_from_cfg(cfg:str) -> nn.ModuleList:
        """ 
        The following functions receives yolo cfg as  list containing dictionaries with the module
        name e.g Convolutional and their respective hyperparameter.
        Args:
            param[in]:list cfg -> list containing parsed cfg file.
            param[out]:nn.ModuleList
        Raises:
            KeyError: If a key doesn't exist in dictionary.
        Returns:
            nn.ModuleList
        """
        warnings.warn('This function will be shifted to yolov3 legacy implementation.')
        if(isinstance(cfg,str)):
            cfg_list = parse_model_config(cfg)
        elif(isinstance(cfg,list)):
            cfg_list = cfg
        else:
            return Exception(f"Provide either path to cfg file or it's list")            
        modules_list = nn.ModuleList()
        parameters = cfg_list.pop(0)
        output_filter = [int(parameters['channels'])]
        #check if the list is empty.
        assert(len(cfg_list) > 0)
        # Add all the layers here which cfgs contain.
        for id , module_dict in enumerate(cfg_list):
            module = nn.Sequential()
            try:
                if(module_dict['type']=='convolutional'):
                    #add Convolutional layer.
                    filters = int(module_dict['filters'])
                    module.add_module((f'Conv:{id}'),conv2d_layer(output_filter[-1],module_dict))

                    #check and  add Batch Norm layer
                    if module_dict['batch_normalize']:
                        module.add_module((f'BatchNorm2d:{id}'),batchNorm2d_layer(filters))

                    #add Activation layer
                    module.add_module((f'Activation:{id}'),activation_layer(module_dict['activation']))
                #Add support for linear layers.    
                elif(module_dict['type']=='upsample'):
                    module.add_module((f'Upsample:{id}'),upsample_layer(int(module_dict['stride'])))
                
                elif(module_dict['type']=='route'):
                    filters = route_layer(module_dict['layers'],output_filter)
                    #Adding an Empty layer in module list.
                    module.add_module((f'Route:{id}'),EmptyLayer())
                
                elif(module_dict['type'] == 'shortcut'):
                    filters = shortcut_layer(int(module_dict['from']),output_filter)
                    #Adding an Empty layer in module list.

                elif(module_dict['type'] == 'yolo'):
                    #reading mask from yolo cfg as index for anchors.
                    anchors_idx = [int(x) for x in  module_dict['mask'].split(',')]
                    #loading anchors inside a list.
                    anchors = [int(x) for x in module_dict['anchors'].split(',')]
                    #list containing tuples of anchors
                    anchors =  [(anchors[i],anchors[i+1]) for i in range(0,len(anchors),2)]
                    #list of tuples of anchors according to the index from cfg file.
                    anchors = [anchors[i] for i in anchors_idx]
                    num_classes = int(module_dict["classes"])
                    img_size = int(parameters['height']) 
                    #Add Yolo layer here
                    yolo_layer = YoloLayer(anchors,num_classes,img_size)
                    module.add_module((f'{id}'),yolo_layer)
                modules_list.append(module)
                output_filter.append(filters)
            
            except KeyError:
                raise KeyError(f"Error key {module_dict['type']} not found")
        return modules_list , output_filter , cfg_list

    @abstractmethod
    def forward_train(self):
        """
        Forward pass function during training.
        """
        raise NotImplementedError
    @abstractmethod
    def forward_test(self):
        """
        Forward pass function during testing.
        """
        raise NotImplementedError
    @abstractmethod
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        """Forward pass function."""
        raise NotImplementedError('Forward Function Not Implemented')
    @abstractmethod
    def load_weights(self,weights:str):
        """
        Initialize the weights of the model.
        Args:
            weights:str weights -> path to weight file  
        Returns:
            nn.Module_list -> nn.module_list containing modules
        """
        #
        raise NotImplementedError('Loading Weights is not implemented for this Model.')
    def save_weights():
        """
        Save weights of the model after checkpoints during training. 
        Should be generic for all classes that are inherited from BaseDetector.
        """
        pass
