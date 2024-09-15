import torch


def bbox_xyhwToxyxy(bbox_xyhw):
    '''
    bbox_xyhw in shape [..., 4]
    height and width of bbox is the full height and width
    '''
    bbox_center = bbox_xyhw[..., :2]
    bbox_hw = bbox_xyhw[..., 2:]
    bbox_hw_half = 0.5 * bbox_hw

    bbox_xyxy = torch.cat([bbox_center - bbox_hw_half, bbox_center + bbox_hw_half], dim=-1)
    return bbox_xyxy


def recover_bbox(bbox, h, w):
    '''
    bbox torch tensor in shape [4] or [...,4], under torch axis
    '''
    bbox_cp = bbox.clone()
    if len(bbox.shape) > 1: # [N,4]
        bbox_cp[..., 0] *= h
        bbox_cp[..., 1] *= w
        bbox_cp[..., 2] *= h
        bbox_cp[..., 3] *= w
        return bbox_cp
    else:
        return torch.tensor([bbox_cp[0]*h, bbox_cp[1]*w, bbox_cp[2]*h, bbox_cp[3]*w])


def generate_anchor_boxes(base_sizes, aspect_ratios, dtype=torch.float32, device='cpu'):
    """
    Generate a set of anchor boxes with different sizes and aspect ratios.

    Arguments:
    base_sizes -- torch.Tensor of shape [N,2], containing N base sizes for the anchor boxes
    aspect_ratios -- torch.Tensor of shape [M], containing M aspect ratios for each base size
    dtype -- the data type of the output tensor
    device -- the device of the output tensor

    Returns:
    anchor_boxes -- torch.Tensor of shape [N*M,4], containing N*M anchor boxes represented as (center_h, center_w, box_h, box_w)
    """

    num_base_sizes = base_sizes.shape[0]
    num_aspect_ratios = aspect_ratios.shape[0]

    # Generate base anchor boxes
    base_boxes = torch.zeros((num_base_sizes * num_aspect_ratios, 4), dtype=dtype, device=device)
    for i in range(num_base_sizes):
        for j in range(num_aspect_ratios):
            w = torch.sqrt(base_sizes[i, 0] * base_sizes[i, 1] / aspect_ratios[j])
            h = aspect_ratios[j] * w
            idx = i * num_aspect_ratios + j
            base_boxes[idx] = torch.tensor([0, 0, h, w], dtype=dtype, device=device)

    return base_boxes


def generate_anchor_boxes_on_regions(
    image_size,
    num_regions,
    base_sizes=torch.tensor([[16, 16], [32, 32], [64, 64], [128, 128]], dtype=torch.float32),
    aspect_ratios=torch.tensor([0.5, 1, 2], dtype=torch.float32),
    dtype=torch.float32,
    device='cpu'
):
    """
    Generate a set of anchor boxes with different sizes and aspect ratios for each region of a split image.

    Arguments:
    image_size -- tuple of two integers, the height and width of the original image
    num_regions -- tuple of two integers, the number of regions in the height and width directions
    aspect_ratios -- torch.Tensor of shape [M], containing M aspect ratios for each base size
    dtype -- the data type of the output tensor
    device -- the device of the output tensor

    Returns:
    anchor_boxes -- torch.Tensor of shape [R^2*N*M,4], containing R^2*N*M anchor boxes represented as (center_h, center_w, box_h, box_w)
    """

    # Calculate the base sizes for each region
    region_size = (image_size[0] / num_regions[0], image_size[1] / num_regions[1])

    # Calculate the anchor boxes for each region
    anchor_boxes = torch.empty((0, 4), dtype=dtype, device=device)
    for i in range(num_regions[0]):
        for j in range(num_regions[1]):
            center_h = (i + 0.5) * region_size[0]
            center_w = (j + 0.5) * region_size[1]
            base_boxes = generate_anchor_boxes(base_sizes, aspect_ratios, dtype=dtype, device=device)
            base_boxes[:, 0] += center_h
            base_boxes[:, 1] += center_w
            anchor_boxes = torch.cat([anchor_boxes, base_boxes], dim=0)

    return anchor_boxes


def calculate_iou(boxes1, boxes2):
    """
    Calculate the IoU between two sets of bounding boxes.

    Arguments:
    boxes1 -- torch.Tensor of shape [...,N,4], containing N bounding boxes represented as [x1, y1, x2, y2]
    boxes2 -- torch.Tensor of shape [...,4], containing a single ground truth box represented as [x1, y1, x2, y2]

    Returns:
    iou -- torch.Tensor of shape [...,N], containing the IoU between each box and the ground truth box
    """

    # Add a new dimension to boxes2 for broadcasting
    boxes2 = boxes2.unsqueeze(-2)    # shape: [...,1,4]

    # Compute the coordinates of the top-left and bottom-right corners of the boxes
    boxes1_tl = boxes1[..., :2]
    boxes1_br = boxes1[..., 2:]
    boxes2_tl = boxes2[..., :2]
    boxes2_br = boxes2[..., 2:]

    # Compute the coordinates of the intersection rectangle
    tl = torch.max(boxes1_tl, boxes2_tl)
    br = torch.min(boxes1_br, boxes2_br)

    # Compute the width and height of the intersection rectangle
    wh = br - tl
    wh[wh < 0] = 0

    # Compute the area of the intersection and union rectangles
    intersection_area = wh[..., 0] * wh[..., 1]
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    union_area = area1 + area2 - intersection_area

    # Compute the IoU between each box and the ground truth box
    iou = intersection_area / union_area

    return iou


def check_bbox(bbox, h, w):   # TODO: too strict, need to relax the invalidity condition
    B, T, _ = bbox.shape
    bbox = bbox.reshape(-1,4)

    y1, x1, y2, x2 = bbox[...,0], bbox[...,1], bbox[...,2], bbox[...,3]
    left_invalid = x2 <= 0.0
    right_invalid = x1 >= w - 1
    top_invalid = y2 <= 0.0
    bottom_invalid = y1 >= h - 1

    y_invalid = torch.logical_or(top_invalid, bottom_invalid)
    x_invalid = torch.logical_or(left_invalid, right_invalid)
    invalid = torch.logical_or(y_invalid, x_invalid)
    valid = ~invalid

    y1_clip = y1.clip(min=0.0, max=h).unsqueeze(-1)
    y2_clip = y2.clip(min=0.0, max=h).unsqueeze(-1)
    x1_clip = x1.clip(min=0.0, max=w).unsqueeze(-1)
    x2_clip = x2.clip(min=0.0, max=w).unsqueeze(-1)
    bbox_clip = torch.cat([y1_clip, x1_clip, y2_clip, x2_clip], dim=-1)

    return bbox_clip.reshape(B,T,4), valid.reshape(B,T)
