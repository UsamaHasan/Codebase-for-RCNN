import sys
import torch
import torch.nn as nn
import numpy as np
from torchvision.ops import nms
#for unit testing
sys.path.append('/home/ncai/RoadSurfaceAnalysis/src')
from detection.models.builder import build_detector
from detection.models.detectors.base import BaseDetector
from detection.utils.config import *
def init_detector(cfg_file=None,checkpoint=None):
    """
    Initialize Model. 
    Args:
        cfg_file(str) : path to cfg file
        checkpoint(str) : path to checkpoint/weights file 
    Returns:
        BaseDetector child class
    """
    if isinstance(cfg_file,str) and not None:

        model = build_detector(cfg_file)
        
        #load checkpoint of the model.
        if checkpoint is not None:
            #call model.load_checkpoint here.
            model.load_weights(checkpoint)
        #check for avaiable devices
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        #set model on evaluation mode.0
        model.eval()
        
        return model
    else:
        raise TypeError('Object type Should be str -> path to model config file')

def inference_detector(detector,img):
    """
    Args:
        detector(BaseDetector) : 
        img (np.ndarray , torch.Tensor):
        Example:
            img -> img.shape(3,100,100)
    Returns:
        output(np.ndarray): 
    """
    
    if isinstance(detector,(BaseDetector,nn.Module)):
        if isinstance(img,(torch.Tensor,np.ndarray)):
            #check if the img is numpy array.
            if isinstance(img,np.ndarray):
                #convert it into torch.tensor
                torch.from_numpy(img)
            #check for channel first.
            if img.size(0) not in [3]:
                #make channel first.
                img = img.permute(2,0,1)
            #append batch_size:
            img = img.view(1,img.size(0),img.size(1),img.size(2))
            with torch.no_grad():
                output = detector(img)
                #Apply non-max suppression
                
                nms(output,torch.tensor(CONFIDENCE_THRESHOLD),NMS_THRESHOLD)
        else:
            raise TypeError(f'Input should be an Image of type np.ndarry or torch.Tensor')
        #
    else:
        raise TypeError(f'Should pass a Detector object')

# for unit testing.
if __name__ == '__main__':

    detector = init_detector('/home/ncai/RoadSurfaceAnalysis/src/cfg/yolov3.cfg',\
        checkpoint='/home/ncai/RoadSurfaceAnalysis/src/weights/yolov3_3300.weights')
    inp = torch.randn((3,416,416),device='cuda')
    inference_detector(detector,inp)