import sys
#sys.path.append('/home/ncai01/Codebase-of-RCNN')
from detection.models.detectors import *
from importlib import import_module
from detection.utils.utils import parse_path
#dictionary containing the names of all model that are implemented.

def build_detector(cfg_file):
    """
    Args:
        cfg_file(str): 
    Returns:
        
    """
    if isinstance(cfg_file,str):
        model_name = parse_path(cfg_file)
        module = import_module(('detection.models.detectors.' + model_name ))
        Detector = getattr(module,model_name.capitalize())
        detector = Detector(cfg_file)
        return detector 
    else:
        raise TypeError('Should provide path to cfg file')
