import os , sys
import torch
import torch.nn as nn
from torch import Tensor

#for unit testing remove after first build.
sys.path.append('/home/ncai/RoadSurfaceAnalysis/src/')
from detection.models.detectors.base import BaseDetector
from detection.utils.config import *

class Yolov3(BaseDetector):
    """
    Implementation of YOLO model for object detection also known as darknet.
    """
    def __init__(self, weights_path = '',cfg_path = ''):
        """ 
        The following function will initializa a torch model, you can either initialize the model and 
        set the weights here. Or can the respective functions later.
        Args:(Optional)
            param[in]: weights_path -> path to weight file to initialize weights.(default='')
            param[in]: cfg_path -> path to cfg file.(default=[])
        """
        super(Yolov3,self).__init__(cfg_path)
        
        if(weights_path!=''):
            super(Yolov3,self)._model_set_weights(weights_path)
        else:
            pass #Implement a method to define trained weights of yolo and initialize model with them
        
            
    def forward(self,x:Tensor,targets=None) -> Tensor:
        """
        Recevices Image as Tensor and return bbox 
        Args:
            param[in]:input Tensor-> Image as input tensor.
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
                x = torch.cat((layers_output[int(layer)] for layer in module_dict['layers'].split(',')),1)
            elif module_dict['type'] == 'yolo':
                #yolo Layer.
                x , layer_loss = module[0](x,targets,input_dim)
                loss += layer_loss
                yolo_output.append(x)
            layers_output.append(x)
        yolo_output = (torch.cat(yolo_output,1)).to('cpu')
        return yolo_output if targets is None else (yolo_output,loss)
        #should raise not Implemented error
    def _init_model():
        """
        private function to initialize model after cfg has initialized.
        """
        pass
    def forward_train():
        """
        """
        raise NotImplementedError
    def forward_test():
        """
        """
        raise NotImplementedError

# For unit testing .
if __name__ == '__main__':

    net = Yolov3()
    temp_input = torch.randn(size=(1,3,416,416))
    output = net(temp_input) # should raise error.
    print(output)