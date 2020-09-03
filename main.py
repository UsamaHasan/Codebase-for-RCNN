import os
import sys
from utils.utils import parse_model_config
from utils.config import *
import torch
from net.yolov3 import Yolov3
if __name__ == '__main__':

    #list of dictionaries containning modules of yolo. 
    module_list = parse_model_config(CFG_PATH)
    #Initialize model.
    model = Yolov3()
    #Initiailize model
    model.model_from_cfg(module_list)