import numpy as np
from torchvision.ops import nms
# Also write implementation in cuda c++ for optimization.
def non_max_suppression(output , confidence_threshold):
    """
    Applies Non Max suppression on the number of predicted boxes.
    https://link.springer.com/chapter/10.1007/978-3-642-17688-3_41
    Args:
        pred_bbox(tensor) : 
    Returns:
        bbox(tensor) : 
    """
    bboxes = output[...,:4]
    prediction_score = output[...,4]
    #squeeze prediction_score tensor
    final_results = nms(bboxes,prediction_score,confidence_threshold)
    return final_results
# Also write implementation in cuda c++ for optimization.
def intersection_over_union(bbox1,bbox2,x1y1x2y2):
    """
    Calculate intersection over union (overlap) between target and 
    predicted bounding boxes.
    Args:
        gt_bbox(tensor) : Ground truth bounding boxes   
        pred_bbox(tensor) : Predicted bounding boxes
    Returns:
        iou(tensor) : intersection over union.
    """
    #Add faster RCNN implementation here.
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = bbox1[:, 0] - bbox1[:, 2] / 2, bbox1[:, 0] + bbox1[:, 2] / 2
        b1_y1, b1_y2 = bbox1[:, 1] - bbox1[:, 3] / 2, bbox1[:, 1] + bbox1[:, 3] / 2
        b2_x1, b2_x2 = bbox2[:, 0] - bbox2[:, 2] / 2, bbox2[:, 0] + bbox2[:, 2] / 2
        b2_y1, b2_y2 = bbox2[:, 1] - bbox2[:, 3] / 2, bbox2[:, 1] + bbox2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = bbox1[:, 0], bbox1[:, 1], bbox1[:, 2], bbox1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = bbox2[:, 0], bbox2[:, 1], bbox2[:, 2], bbox2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou
    

def build_targets(pred_bbox,target_bbox,anchors,pred_classes,confidence_threshold):
    """
    Calculates intersection_over_union and applies non_max_suppression.
    Args:   
        pred_bbox(tensor) : 
        target_bbox(tensor):
        anchors(tensor):
        pred_classes(tensor):
        confidence_threshold(int):
    Returns:
        final_pred_bbox(tensor):
    """
    raise NotImplementedError(f'functionality not implemented')
