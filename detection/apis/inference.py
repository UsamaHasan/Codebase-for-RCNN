import sys
#for unit testing
sys.path.append('/home/ncai01/Codebase-of-RCNN')
from detection.models.builder import build_detector
def init_detector(cfg_file:str):
    """
    Initialize Model. 
    Args:

    Returns:
    """
    if isinstance(cfg_file,str):
        detector = build_detector(cfg_file)
        return detector
    else:
        raise TypeError('Object type Should be str -> path to model config file')

def inference_detector():
    """
    Args:
    Returns:
    """
    pass
if __name__ == '__main__':
    init_detector('/home/ncai01/Codebase-of-RCNN/cfg/yolov3.cfg')