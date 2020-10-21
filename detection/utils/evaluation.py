"""
This file will contain all the metrics related to the evaluation of the detection models.
"""
import torch
import numpy as np


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def average_precision(ious,recall,precision,):
    """
    Calculates Minimum Average Precision.
    Average Precision (AP) is finding the area under the precision-recall curve.
    Implement For Pascal VOC 2007 , VOC 2012, and COCO Dataset.
    Formula:
    """
    # % Ap at IOU=0.50
    # Smooth the zigzag curve.
    i = 0
    while(i < len(precision)):
        if precision[i] < precision[i+1]:
            precision[i] = precision[i+1]
    recall_ = np.linspace(0,1,11)
    #Consider them n rows and 2 cols matrix.
    np.where()



if __name__ ==  '__main__':
    recall = np.linspace(0,1,5)
    precision = [0.1,0.2,0.3,0.4,0.6,0.7]
    print(f'Recall:{recall}')
    print(f'Precision:{precision}')
    eval = ElevenPointInterpolatedAP(recall,precision)
    print(eval)
    