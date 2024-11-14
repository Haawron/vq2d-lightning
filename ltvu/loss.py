from einops import rearrange, repeat
import torch
import torch.nn.functional as F

from ltvu.bbox_ops import calculate_iou


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0):
    '''
    focal loss for binary classification (background/foreground)
    inputs and targets in shape [N]
    inputs are not activated by sigmoid
    alpha is the weight for negatives (background)
    '''
    targets = targets.float()
    device = targets.device

    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

    pt = torch.sigmoid(inputs)
    pt = torch.where(targets == 1, pt, 1 - pt)

    alpha = torch.where(targets == 1, 1 - alpha, alpha).to(device)

    F_loss = alpha * (1 - pt)**gamma * BCE_loss

    #F_loss = alpha * BCE_loss

    return F_loss.mean()


def GiouLoss(bbox_p, bbox_g, mask=None):
    """
    :param bbox_p: predict of bbox(N,4)(x1,y1,x2,y2)
    :param bbox_g: groundtruth of bbox(N,4)(x1,y1,x2,y2)
    :param mask: ground truth of valid instance, in shape [B]
    :return:
    """
    device = bbox_p.device
    # for details should go to https://arxiv.org/pdf/1902.09630.pdf
    # ensure predict's bbox form
    x1p = torch.minimum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    x2p = torch.maximum(bbox_p[:, 0], bbox_p[:, 2]).reshape(-1,1)
    y1p = torch.minimum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)
    y2p = torch.maximum(bbox_p[:, 1], bbox_p[:, 3]).reshape(-1,1)

    bbox_p = torch.cat([x1p, y1p, x2p, y2p], axis=1)
    # calc area of Bg
    area_p = (bbox_p[:, 2] - bbox_p[:, 0]) * (bbox_p[:, 3] - bbox_p[:, 1])
    # calc area of Bp
    area_g = (bbox_g[:, 2] - bbox_g[:, 0]) * (bbox_g[:, 3] - bbox_g[:, 1])

    # cal intersection
    x1I = torch.maximum(bbox_p[:, 0], bbox_g[:, 0])
    y1I = torch.maximum(bbox_p[:, 1], bbox_g[:, 1])
    x2I = torch.minimum(bbox_p[:, 2], bbox_g[:, 2])
    y2I = torch.minimum(bbox_p[:, 3], bbox_g[:, 3])
    I = torch.maximum((y2I - y1I), torch.tensor([0.0]).to(device)) * torch.maximum((x2I - x1I), torch.tensor([0.0]).to(device))

    # find enclosing box
    x1C = torch.minimum(bbox_p[:, 0], bbox_g[:, 0])
    y1C = torch.minimum(bbox_p[:, 1], bbox_g[:, 1])
    x2C = torch.maximum(bbox_p[:, 2], bbox_g[:, 2])
    y2C = torch.maximum(bbox_p[:, 3], bbox_g[:, 3])

    # calc area of Bc
    area_c = (x2C - x1C) * (y2C - y1C)
    U = area_p + area_g - I
    iou = 1.0 * I / (U + 1e-6)

    # Giou
    giou = iou - (area_c - U) / (area_c + 1e-6)
    if not torch.isfinite(giou).all():
        print('giou is not finite')
        giou = torch.zeros_like(giou, device=device, requires_grad=True, dtype=torch.float32)

    if torch.is_tensor(mask):
        loss_giou = torch.mean(1.0 - giou[mask])
    else:
        loss_giou = torch.mean(1.0 - giou)
    return iou, giou, loss_giou


def process_labels(labels, iou, topk=10):
    '''
    labels: in shape [B,T,N], bool
    iou: in shape [B,T,N]
    '''
    B,T,N = labels.shape

    labels = rearrange(labels, 'b t n -> (b t n)')
    iou = rearrange(iou, 'b t n -> (b t n)')

    if not labels.any():
        # no pos assigned, choose topk anchors with largest iou as positives
        _, topk_indices = torch.topk(iou, k=topk)
        labels[topk_indices] = True

    labels = rearrange(labels, '(b t n) -> b t n', b=B, t=T, n=N)
    return labels


def assign_labels(anchors, gt_boxes, iou_threshold=0.5, topk=5):
    """
    Assign labels to a set of bounding box proposals based on their IoU with ground truth boxes.

    Arguments:
    anchors -- torch.Tensor of shape [B,T,N,4], representing the bounding box proposals for each frame in each clip
    gt_boxes -- torch.Tensor of shape [B,T,4], representing the ground truth boxes for each frame in each clip
    iou_threshold -- float, the IoU threshold for a proposal to be considered a positive match with a ground truth box

    Returns:
    labels -- torch.Tensor of shape [B,T,N], containing the assigned labels for each proposal (0 for background, 1 for object)
    """
    anchors = anchors.detach()
    gt_boxes = gt_boxes.detach()

    b,t = gt_boxes.shape[:2]    #[B,T,N,4]

    # Calculate the IoU between each proposal and the ground truth box
    iou = calculate_iou(anchors.view(-1, anchors.shape[-2], anchors.shape[-1]),   # [B*T,N,4]
                        gt_boxes.view(-1, gt_boxes.shape[-1]))                    # [B*T,4] -> [B*T,N]
    iou = iou.view(anchors.shape[:-1])    # [B,T,N]

    # Assign labels to the proposals based on their IoU with the ground truth box
    labels = iou > iou_threshold

    if not labels.any():
        labels = process_labels(labels, iou, topk)

    return labels


def get_losses_with_anchor(
    preds, gts,
    training = True,
    use_hnm: bool = False,
    positive_threshold = .2,
    positive_topk = 5,
    weight_bbox_center = 1.,
    weight_bbox_hw = 1.,
    weight_bbox_giou = .3,
    weight_prob = 100.,
):
    if use_hnm:
        gts = replicate_sample_for_hnm(gts)

    pred_center = preds['center']   # [b,t,N,2]
    pred_hw = preds['hw']           # [b,t,N,2], actually half of hw
    pred_bbox = preds['bbox']       # [b,t,N,4]
    pred_prob = preds['prob']       # [b,t,N]
    anchor = preds['anchor']        # [1,1,N,4]
    b,t,N = pred_prob.shape
    device = pred_prob.device

    if 'center' not in gts.keys():
        gts['center'] = (gts['clip_bbox'][...,:2] + gts['clip_bbox'][...,2:]) / 2.0
    if 'hw' not in gts.keys():
        gts['hw'] = gts['center'] - gts['clip_bbox'][...,:2]   # actually half of hw
    gt_center = gts['center']               # [b,t,2]
    gt_hw = gts['hw']                       # [b,t,2]
    gt_bbox = gts['clip_bbox']              # [b,t,4]
    gt_prob = gts['clip_with_bbox']         # [b,t]
    gt_before_query = gts['before_query']   # [b,t]

    # assign labels to anchors
    if training and gt_prob.bool().any():
        if list(anchor.shape[:2]) == [b,t]:
            temp_anchor = anchor
        else:
            temp_anchor = anchor.repeat(b,t,1,1)
        assign_label = assign_labels(temp_anchor, gt_bbox,   # anchor.repeat(b,t,1,1) / pred_bbox
                                     iou_threshold=positive_threshold,
                                     topk=positive_topk)               # [b,t,N]
        positive = torch.logical_and(gt_prob.unsqueeze(-1).repeat(1,1,N).bool(),
                                     assign_label.bool())                           # [b,t,N]
        positive = rearrange(positive, 'b t N -> (b t N)')                          # [b*t*N]
    elif not training:
        positive = repeat(gt_prob.bool(), 'b t -> (b t N)', N=N)
    else:
        positive = torch.zeros(b,t,N).reshape(-1).bool().to(device)

    if torch.sum(positive.float()).item() == 0:
        positive[:1] = True
    loss_mask = positive.float().unsqueeze(1)                                    # [b*t*N,1]
    # DEBUG: why didn't mask before_query here?

    # anchor box regression loss
    if torch.sum(positive.float()).item() > 0:
        # bbox center loss
        pred_center = rearrange(pred_center, 'b t N c -> (b t N) c')
        pred_center_positive = pred_center[positive.bool()]
        gt_center_positive = rearrange(gt_center.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')[positive.bool()]
        loss_center = F.l1_loss(pred_center_positive, gt_center_positive)

        # bbox hw loss
        pred_hw = rearrange(pred_hw, 'b t N c -> (b t N) c')
        pred_hw_positive = pred_hw[positive.bool()]
        gt_hw_positive = rearrange(gt_hw.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c')[positive.bool()]
        loss_hw = F.l1_loss(pred_hw_positive, gt_hw_positive)

        # bbox giou loss
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t N) c').float()
        gt_bbox_replicate = rearrange(gt_bbox.unsqueeze(2).repeat(1,1,N,1), 'b t N c -> (b t N) c').float()
        iou, giou, loss_giou = GiouLoss(pred_bbox, gt_bbox_replicate, mask=loss_mask.bool().squeeze())
    else:
        pred_bbox = rearrange(pred_bbox, 'b t N c -> (b t N) c')
        req = pred_bbox.requires_grad
        loss_center = torch.tensor(0., requires_grad=req).cuda()
        loss_hw = torch.tensor(0., requires_grad=req).cuda()
        loss_giou = torch.tensor(0., requires_grad=req).cuda()
        iou = torch.tensor(0., requires_grad=req).cuda()
        giou = torch.tensor(0., requires_grad=req).cuda()

    # if use_hnm:
    #     loss_prob = BCELogitsLoss_with_HNM(pred_prob, gt_prob, positive, gt_before_query, [1.0, 1.0])
    #     pred_prob = rearrange(pred_prob, 'b t N -> (b t N)')
    # else:
    pred_prob = rearrange(pred_prob, 'b t N -> (b t N)')
    gt_before_query_replicate = rearrange(gt_before_query.unsqueeze(2).repeat(1,1,N), 'b t N -> (b t N)')
    loss_prob = focal_loss(pred_prob[gt_before_query_replicate.bool()].float(),
                        positive[gt_before_query_replicate.bool()].float())

    loss = {
        'loss_bbox_center': loss_center,
        'loss_bbox_hw': loss_hw,
        'loss_bbox_giou': loss_giou,
        'loss_prob': loss_prob,

        # weights
        'weight_bbox_center': weight_bbox_center,
        'weight_bbox_hw': weight_bbox_hw,
        'weight_bbox_giou': weight_bbox_giou,
        'weight_prob': weight_prob,

        # information
        'iou': iou.detach(),
        'giou': giou.detach(),
    }

    # get top prediction
    pred_prob = rearrange(pred_prob, '(B N) -> B N', N=N)                     # [b*t,N]
    pred_bbox = rearrange(pred_bbox, '(B N) c -> B N c', N=N)                 # [b*t,N,4]
    pred_prob_top, top_idx = torch.max(pred_prob, dim=-1)                     # [b*t], [b*t]
    top_idx = repeat(top_idx, 'B -> B n c', n=1, c=4)  # DEBUG: 에휴
    pred_bbox_top = torch.gather(pred_bbox, dim=1, index=top_idx).squeeze()   # [b*t,4]

    pred_top = {
        'bbox': rearrange(pred_bbox_top, '(b t) c -> b t c', b=b, t=t),
        'prob': rearrange(pred_prob_top, '(b t) -> b t', b=b, t=t)
    }

    return loss, pred_top, gts, positive  # gts with hw, center computed


def replicate_sample_for_hnm(gts):
    '''
        gts = {
            'clip':                 in [b,t,c,h,w]
            'clip_rigin':           in [b,t,c,h,w]
            'clip_with_bbox':       in [b,t]
            'before_query':         in [b,t]
            'clip_bbox':            in [b,t,4]
            'query':                in [b,c,h,w]
            'query_origin':         in [b,c,h,w]
            'clip_h':               in [b]
            'clip_w':               in [b]
        }
    '''
    # clip = gts['clip']
    # clip_origin = gts['clip_origin']
    clip_with_bbox = gts['clip_with_bbox']
    before_query = gts['before_query']
    clip_bbox = gts['clip_bbox']
    # query = gts['query']
    # query_origin = gts['query_origin']
    # clip_h, clip_w = gts['clip_h'], gts['clip_w']

    b, t = before_query.shape[:2]
    device = before_query.device

    # new_clip = []
    # new_clip_origin = []
    new_clip_with_bbox = []
    new_before_query = []
    new_clip_bbox = []
    # new_query = []
    # new_query_origin = []
    # new_clip_h, new_clip_w = [], []

    for i in range(b):
        for j in range(b):
            # new_clip.append(clip[i])
            # new_clip_origin.append(clip_origin[i])
            # new_query.append(query[j])
            # new_query_origin.append(query_origin[j])
            if i == j:
                new_clip_with_bbox.append(clip_with_bbox[i])
                new_before_query.append(before_query[i])
                new_clip_bbox.append(clip_bbox[i])
            else:
                new_clip_with_bbox.append(torch.zeros(t).float().to(device))
                new_before_query.append(torch.ones(t).bool().to(device))
                new_clip_bbox.append(torch.tensor([[0.0, 0.0, 0.0001, 0.0001]]).repeat(t,1).float().to(device))
            # new_clip_h.append(clip_h[i])
            # new_clip_w.append(clip_w[i])
    
    # new_clip = torch.stack(new_clip)
    # new_clip_origin = torch.stack(new_clip_origin)
    new_clip_with_bbox = torch.stack(new_clip_with_bbox)
    new_before_query = torch.stack(new_before_query)
    new_clip_bbox = torch.stack(new_clip_bbox)
    # new_clip_h = torch.stack(new_clip_h)
    # new_clip_w = torch.stack(new_clip_w)
    # new_query = torch.stack(new_query)
    # new_query_origin = torch.stack(new_query_origin)

    new_gts = {
            # 'clip': new_clip,                       # in [b^2,t,c,h,w]
            # 'clip_origin': new_clip_origin,         # in [b^2,t,c,h,w]
            'clip_with_bbox': new_clip_with_bbox,   # in [b^2,t]
            'before_query': new_before_query,       # in [b^2,t]
            'clip_bbox': new_clip_bbox,             # in [b^2,t,4]
            # 'query': new_query,                     # in [b^2,c,h,w]
            # 'query_origin': new_query_origin,       # in [b^2,c,h,w]
            # 'clip_h': new_clip_h,                   # in [b^2]
            # 'clip_w': new_clip_w,                   # in [b^2]
        }
    return new_gts


def BCELogitsLoss_with_HNM(pred_prob, gt_prob, positive, gt_before_query, weight):
    '''
    pred_prob: predicted probability of anchors, in shape [b,t,N], without sigmoid
    gt_prob: GT probability of frames, in shape [b,t]
    positive: assigned labels of anchors, in shape [b*t*N]
    gt_before_query: mask for frames before query frame, in shape [b,t]
    weight: weights for positive and negative predictions
    '''
    b,t,N = pred_prob.shape
    gt_prob = gt_prob.unsqueeze(-1).repeat(1,1,N)   # [b,t,N]

    pred_prob = rearrange(pred_prob, 'b t N -> (b t N)')                                # [b*t*N]
    gt_prob = rearrange(gt_prob, 'b t N -> (b t N)')                                    # [b*t*N]
    BCE_loss = F.binary_cross_entropy_with_logits(pred_prob, gt_prob, reduction='none') # [b*t*N]

    pred_prob = rearrange(pred_prob, '(b t N) -> b t N', b=b, t=t)
    gt_prob = rearrange(gt_prob, '(b t N) -> b t N', b=b, t=t)
    BCE_loss = rearrange(BCE_loss, '(b t N) -> b t N', b=b, t=t)
    positive = rearrange(positive, '(b t N) -> b t N', b=b, t=t)

    loss = HardNegMining(pred_prob, gt_prob, positive, BCE_loss, gt_before_query, weight)
    return loss.mean()


def HardNegMining(pred_prob, gt_prob, positive, BCE_loss, gt_before_query, weight, ratio_neg_pos=3., ratio_hard=0.05):
    '''
    Perform frame-level hard negative mining
    Params:
        ratio_neg_pos: negative / positive ratio
        ratio_hard: ratio of negatives from all anchors if no positive anchor is assigned
        pred_prob, gt_prob, positive, BCE_loss in [b,t,N]
        gt_before_query: in [b,t]
        weight: weights for positive and negative predictions
    Mine the anchor boxes with three type:
        1. query object doesn't occur and no anchor is assigned as positive
        2. query object occurs and some anchors are assigned as positive
        3. query object occurs but no anchor is assigned as positive
    '''
    b, t, N = pred_prob.shape
    b_real = int(b ** 0.5)      # for example, real batch size B will lead to B^2 training batch leveraging cross-video data
    w_pos, w_neg = weight

    mined_loss = []
    for i in range(b_real):
        # get results for each visual query
        query_idx = [(i + j * b_real) for j in range(b_real)]       # corresponds to how we generate cross-video data

        cur_gt_before_query = gt_before_query[query_idx].bool()     # [b_real, t]
        
        cur_positive = positive[query_idx][cur_gt_before_query]     # [M], for all valid anchor box of the query (reject unreliable ones after query time)
        cur_loss = BCE_loss[query_idx][cur_gt_before_query]         # [M]
        M = cur_loss.shape[0]

        num_pos = int(torch.sum(cur_positive).item())
        num_neg = int(ratio_neg_pos * num_pos) if num_pos > 0 else int(ratio_hard * M)

        cur_loss_positives = cur_loss[cur_positive.bool()]          
        cur_loss_negatives = cur_loss[~cur_positive.bool()]
        num_neg = num_neg if num_neg < cur_loss_negatives.shape[0] else cur_loss_negatives.shape[0]
        cur_loss_negatives_hard, _ = torch.topk(cur_loss_negatives, num_neg)

        mined_loss.append(cur_loss_positives * w_pos)
        mined_loss.append(cur_loss_negatives_hard * w_neg)
    
    mined_loss = torch.cat(mined_loss, dim=0)
    return mined_loss