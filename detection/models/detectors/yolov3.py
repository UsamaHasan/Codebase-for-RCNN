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
        
            
    def forward(self,input:Tensor) -> Tensor:
        """
        Recevices Image as Tensor and return bbox 
        Args:
            param[in]:input Tensor-> Image as input tensor.
        """
        print(self.modules_list)
        output = super(Yolov3,self).forward(input)
        return output
        #should raise not Implemented error
    def _init_model():
        """
        private function to initialize model after cfg has initialized.
        """
        pass
    def forward_train():
        """
        """
        pass
    def forward_test():
        """
        """
        pass

# For unit testing .
if __name__ == '__main__':

    net = Yolov3()
    temp_input = torch.randn(size=(100,100))
    output = net(temp_input) # should raise error.
    print(output)