import os , sys
import torch
import torch.nn as nn
from torch import Tensor

#for unit testing remove after first build.
sys.path.append('/home/ncai01/Codebase-of-RCNN/')
from detection.models.detectors.base import BaseDetector
from detection.utils.config import *
from detection.models.utils.model_visualization import viz
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
            weights_path(str): path to weight file to initialize weights.(default='')
            cfg_path(str): path to cfg file.(default=[])
        """
        if(cfg_path!=''):
            super(Yolov3,self).__init__(cfg_path)
        else:
            super(Yolov3,self).__init__(YOLO_V3_CFG_PATH)
        if(weights_path!=''):
            super(Yolov3,self)._model_load_weights(weights_path)
            
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

        Return:
        """
        super(Yolov3,self).load_weights(weight_file)

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

# For unit testing .
if __name__ == '__main__':
    net = Yolov3()
    inp = torch.randn(size=(1,3,416,416))
    output = net(inp)
    print(output.shape)