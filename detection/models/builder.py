import sys
sys.path.append('/home/ncai01/Codebase-of-RCNN')
from detection.models.detectors import *
from importlib import import_module
from detection.utils.utils import parse_path
#dictionary containing the names of all model that are implemented.
models_dict = {'yolov3':'yolov3'}

def build_detector(cfg_file):
    """
    Args:
        cfg_file(str): 
    Returns:
    """
    if isinstance(cfg_file,str):
        model_name = parse_path(cfg_file)
        module_name = models_dict[model_name]
        module = import_module(('detection.models.detectors.' + module_name))
        Detector = getattr(module,'Yolov3')
        detector = Detector(cfg_file)
        return detector 
    else:
        raise TypeError('Should provide path to cfg file')

# for unit testing.
if __name__ == '__main__':
    build_detector('/home/ncai01/Codebase-of-RCNN/cfg/yolov3.cfg')