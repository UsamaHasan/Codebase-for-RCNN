import numpy as np
import torch
from torchvision.ops import nms
from PIL import Image , ImageDraw
# Also write implementation in cuda c++ for optimization.

def non_max(bbox,confidence_threshold,nms_thres):
    """
    """
    
    bbox = torch.squeeze(bbox)
    
    boxes = xywh2xyxy(bbox[:,:4])
    bbox = bbox[bbox[:, 4] >= confidence_threshold]
    if bbox.numel() ==0:
        return bbox
    score = bbox[:, 4] * bbox[:, 5:].max(1)[0]

    # Sort by it
    bbox = bbox[(-score).argsort()]
    class_confs, class_preds = bbox[:, 5:].max(1, keepdim=True)
    
    detections = torch.cat((bbox[:, :5], class_confs.float(), class_preds.float()), 1)
    # Perform non-maximum suppression

    indexes = nms(boxes,score,nms_thres)
    #Select the indexes with heighest ratio from bbox tensor
    #boxes = detections[indexes]
    boxes = detections[indexes[:1]]
    boxes = boxes[boxes[:,4] >= confidence_threshold]
    
    return boxes


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
        img = img.detach().cpu().numpy()
    if isinstance(detections,torch.Tensor):
        detections = detections.detach().cpu().numpy()
    #detections = rescale_boxes(detections, img.shape[2], (416,416))
    unique_labels = np.unique(detections[:,-1])
    n_cls_preds = len(unique_labels)
    img = img*255
    img = img.reshape(img.shape[2],img.shape[3],img.shape[1])
    img = Image.fromarray(np.uint8(img))
    
    if detections is not None:
        draw_img = ImageDraw.Draw(img)
    
    #Error detections only contain 4 values 
    print(detections.shape)
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        #print(f'bbox coordinates  {x1} {y1} {x2} {y2}')
        cordinates = [(x1,y1),(x2,y2)]
        draw_img.rectangle(cordinates,fill=None,outline='red')
    return [img,conf,cls_pred] 


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y
