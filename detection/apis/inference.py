import sys
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from detection.models.builder import build_detector
from detection.models.detectors.base import BaseDetector
from detection.utils.config import *
from detection.models.utils.utils import non_max_suppression , draw_bbox
import pdb
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
                #make channel first.
                img = img.permute(2,0,1)
            #append batch_size:
            img = torch.unsqueeze(img,0)
            img = img/255           
            #breakpoint()
            with torch.no_grad():
                detections = detector(img)
                #Apply non-max suppression
                
                detections = torch.squeeze(detections)
                #This function is currently broken and is causing strange behaviour 
                # you can check the function implementation for further clarificiation.
                # Update the function according to 
                bbox = non_max_suppression(detections,CONFIDENCE_THRESHOLD)
                
                #Create function to draw bounding boxes
                output =  draw_bbox(img,bbox) 
                              
                return output
        else:
            raise TypeError(f'Input should be an Image of type np.ndarry or torch.Tensor')
        #
    else:
        raise TypeError(f'Should pass a Detector object')

# for unit testing.
if __name__ == '__main__':
   
    detector = init_detector('/home/ncai01/Codebase-of-RCNN/cfg/yolov3.cfg',\
        checkpoint='/home/ncai01/Codebase-of-RCNN/weights/yolov3-obj_17400.weights')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #input = torch.randn((3,416,416),device=device)
    img = Image.open('/home/ncai01/Downloads/Chiba_20170913105752.jpg')
    img = img.resize((416,416))
    out   = inference_detector(detector,img)