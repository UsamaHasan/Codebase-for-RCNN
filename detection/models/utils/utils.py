import numpy as np
from torchvision.ops import nms
import matplotlib.pyplot as plt

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

def load_state_dict_from_url(url):
    """

    """
    #First download the state file from url.
    #Then call torch.load to load the .pth/.pt file and load 
    pass



def draw_bbox(img,detection):
    """
    """
    detections = rescale_boxes(detections, img.shape[1], img.shape[:2])
    unique_labels = detections[:, -1].cpu().unique()
    n_cls_preds = len(unique_labels)
    bbox_colors = random.sample(colors, n_cls_preds)
    for x1, y1, w, h, conf, cls_conf, cls_pred in detections:

        color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
        # Create a Rectangle patch
        bbox = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor="none")
        # Add the bbox to the plot
        ax.add_patch(bbox)

def rescale_boxes(detections,width,height):
    
    if isinstance(detections,torch.Tensor):
        detections = detections.detach().numpy()
    
    #rescale boxes to original image width and height.
    detections = detections[:,0:4] * np.array([width,height,width,height])

    # Yolo returns box coordinates as center X, center Y and width height for each box
    # x1 = centerx - width/2
    detections[:,0] = detections[:,0] - detections[:,2]/2
    # y1 = centery - height/2
    detections[:,1] = detections[:,1] - detections[:,3]/2

    return detections