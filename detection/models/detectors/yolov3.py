import os , sys
sys.path.append(os.path.join(os.getcwd(),'net/'))

import torch
import torch.nn as nn
from torch import Tensor
from yolo_utils import *
class Yolov3(nn.Module):
    """
    Implementation of YOLO model for object detection also known as darknet.
    """
    def __init__(self, weights_path = '',cfg_list = []):
        """
        The following function will initializa a torch model, you can either initialize the model and 
        set the weights here. Or can the respective functions later.
        Args:(Optional)
            param[in]: weights_path -> path to weight file to initialize weights.(default='')
            param[in]: cfg_path -> cfg module list.(default=[])
        """
        super(Yolov3,self).__init__()
        
        if(weights_path!=''):
            self.model_set_weights(weights_path)
        if(len(cfg_list)!=0):
            self.model_from_cfg(cfg_file)

    def forward(self,input:Tensor) -> Tensor:
        """
        Recevices Image as Tensor and return bbox 
        Args:
            param[in]:input Tensor-> Image as input tensor.
        """
        pass
    def model_from_cfg(self,cfg_list:list) -> nn.ModuleList:
        """ 
        The following functions receives yolo cfg as  list containing dictionaries with the module
        name e.g Convolutional and their respective hyperparameter.
        Args:
            param[in]:list cfg -> list containing parsed cfg file.
            param[out]:nn.ModuleList
        Raises:
            KeyError: If a key doesn't exist in dictionary.
        """
        #contains all the nn.Sequential modules.
        modules_list = nn.ModuleList()
        parameters = cfg_list.pop(0)
        output_filter = [int(parameters['channels'])]
        #check if the list is empty.
        assert(len(cfg_list) > 0)
        
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
                    pass
                modules_list.append(module)
                output_filter.append(filters)
            
            except KeyError:
                raise KeyError(f"Error key {module_dict['type']} not found")
        print('for unit testing.',modules_list)
        return modules_list


    def _model_set_weights(self,weights:str):
        """
        Initialize the weights of the model.
        Args:
            param[in]:str weights -> path to weight file
            param[out]:nn.Module_list -> nn.module_list containing modules  
        """
        pass
    def _init_model():
        """
        private function to initialize model after cfg has initialized.
        """
        pass
