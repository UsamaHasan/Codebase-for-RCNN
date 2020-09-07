import os , sys
import torch
import torch.nn as nn
sys.path.append('/home/ncai/RoadSurfaceAnalysis/src/')
from detection.models.detectors.base import BaseDetector

from torch import Tensor
class Yolov3(BaseDetector):
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
            super(Yolov3,self)._model_set_weights(weights_path)
        if(len(cfg_list)!=0):
            super(Yolov3,self).model_from_cfg()

    def forward(self,input:Tensor) -> Tensor:
        """
        Recevices Image as Tensor and return bbox 
        Args:
            param[in]:input Tensor-> Image as input tensor.
        """
        output = super(Yolov3,self).forward(input)
        return output
        #should raise not Implemented error
    def _init_model():
        """
        private function to initialize model after cfg has initialized.
        """
        pass
    def forward_train():
        pass
    def forward_test():
        pass
if __name__ == '__main__':
    net = Yolov3()
    temp_input = torch.randn(size=(100,100))
    output = net(temp_input) # should raise error.
    print(output)