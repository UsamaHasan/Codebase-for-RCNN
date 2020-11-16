import numpy as np
import torch
from torchvision.ops import nms
import matplotlib.pyplot as plt
#import cv2
# Also write implementation in cuda c++ for optimization.
def non_max_suppression(prediction , confidence_threshold,nms_thres):
    """
    NMS Implementation taken from https://github.com/eriklindernoren/PyTorch-YOLOv3/    
    Args:
        pred_bbox(tensor) : 
    Returns:
        bbox(tensor) : 
    """
    # Have to Implement own NMS
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= confidence_threshold]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            ouptut[image_i] = torch.stack(keep_boxes)
    
    return np.array(output)


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

def load_state_dict_from_url(url):
    """

    """
    #First download the state file from url.
    #Then call torch.load to load the .pth/.pt file and load 
    pass



def draw_bbox(img,detections):
    """
    This functions draws the rectangular boxes on the detected area and returns a list containing

    """
    if isinstance(img,torch.Tensor):
        img = img.detach().numpy()
    if isinstance(detections,torch.Tensor):
        detections = detections.detach().numpy()

    detections = rescale_boxes(detections, img.shape[2], img.shape[3])
    #unique_labels = detections[:, -1].unique()
    unique_labels = np.unique(detections[:,-1])
    n_cls_preds = len(unique_labels)
    #bbox_colors = random.sample(colors, n_cls_preds)

    #Error detections only contain 4 values 
    for x1, y1, w, h, conf, cls_conf, cls_pred in detections:
    #    img = cv2.rectangle(img,(x1,y1),(x1+w,y1+h))
        return [img,conf,cls_conf,cls_pred] 
        #draw boxes with opencv over here.
        #continue

def rescale_boxes(detections,width,height):
    """
    The functions rescales the boxes given as (center_x ,center_y,w,h) and return 
    rescaled coordinates converted to x1,y1,x2,y2.
    """
    #rescale boxes to original image width and height.
    
    detections = detections[:,0:4] * np.array([width,height,width,height])
    return detections

def xywh2xyxy(detections):
    # Yolo returns box coordinates as center X, center Y and width height for each box
    # x1 = centerx - width/2
    detections[:,0] = detections[:,0] - detections[:,2]/2
    # y1 = centery - height/2
    detections[:,1] = detections[:,1] - detections[:,3]/2
    #x2 = x1 + width 
    detections[:,2] = detections[:,0] + detections[:,2]
    #y2 = y1 + height
    detections[:,3] = detections[:,1] + detections[:,3]
    return detections