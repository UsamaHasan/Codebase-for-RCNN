import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from detection.models.builder import build_detector
from detection.models.detectors.base import BaseDetector
from detection.utils.config import *
from detection.models.utils.utils import non_max , draw_bbox ,non_max
from PIL import Image , ImageDraw
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

        #set model on evaluation mode.
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
        output(PIL.Image): 
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(detector,(BaseDetector,nn.Module)):
        if isinstance(img,(torch.Tensor,np.ndarray,Image.Image)):
            #Check for PIL Image.
            if isinstance(img,Image.Image):
                img = np.asarray(img)
            
            #check if the img is numpy array.
            if isinstance(img,np.ndarray):
                #convert it into torch.tensor
                img = torch.from_numpy(img).float().to(device)
            
            #check for channel first.
            if img.size(0) not in [3,2,1]:
                #make channel first .
                #Change this line with a better function.
                img = img.view(img.size(2),img.size(0),img.size(1))

            if img.size(1) != detector.input_height  or img.size(2) != detector.input_width:
                raise  Exception(f'Input Shape does not match the model input.({detector.input_height},{detector.input_width})')     

            #append batch_size:
            img = torch.unsqueeze(img,0)
            # This Line needs to be replaced. Further We need to add normalization of input
            img = img/127.5           
            #breakpoint()
            with torch.no_grad():
                detections = detector(img)
                #Apply non-max suppression
                bbox = non_max(detections,CONFIDENCE_THRESHOLD,NMS_THRESHOLD)
                if bbox.numel() == 0:
                    return None 
                else:
                    output =  draw_bbox(img,bbox)
                return output
        else:
            raise TypeError(f'Input should be an Image of type np.ndarry or torch.Tensor')
        #
    else:
        raise TypeError(f'Should pass a Detector object')

# for unit testing.
if __name__ == '__main__':
    detector = init_detector('/home/ncai/Projects/ApiTesting/yolov3.cfg',\
        checkpoint='/home/ncai/Downloads/yolov3-wider_16000.weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open('/home/ncai/Downloads/1*DW3-mBLhOOAFIFUVYUkgsQ.png')
    img = img.resize((416,416))
    out = inference_detector(detector,img)
    if out is not None:
        out[0].show()