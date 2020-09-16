import os
"""
This contains all the paths, keys and other information neccessary

"""
YOLO_V3_CFG_PATH = os.path.join('/home/ncai/RoadSurfaceAnalysis/src' ,'cfg/yolov3.cfg') 

CONVOLUTIONAL = ['convolutional','Convolutional']
BATCH_NORMALIZATION = ['batch_normalize','Batch_Norm','batch_norm','batch_normalization']

NMS_THRESHOLD = 0.4
CONFIDENCE_THRESHOLD = 0.8