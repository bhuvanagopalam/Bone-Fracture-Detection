import torch
import torchvision.ops.boxes as box_ops
import numpy as np

from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.detection.iou import IntersectionOverUnion

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2],1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], 1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[ :,0]) * (a[:, 3] - a[:,1]),1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


def calculate_iou(box1, box2):

    
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - intersection_area
    
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou






def calculate_recall_precision(
    ground_truths,
    predictions,
    scores,
    iou_threshold=0.5,
):

    true_positives = []
    false_positives = []
    all_scores = []
    num_ground_truths = len(ground_truths)

    for gt, pred, score in zip(ground_truths, predictions, scores):
        iou = calculate_iou(gt, pred)
        all_scores.append(score.detach().numpy())

        if iou >= iou_threshold:
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    all_scores = np.array(all_scores)

    indices = np.argsort(-all_scores)
    true_positives = true_positives[indices]
    false_positives = false_positives[indices]

    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)

    precision = true_positives / (true_positives + false_positives + np.finfo(float).eps)
    recall = true_positives / (num_ground_truths + np.finfo(float).eps)



    return np.mean(precision), np.mean(recall)

def compute_metrics(model, loader):
    IOU = 0
    num_pos = 0

    map_50 = 0
    predicted_box = []
    annots = []
    scores_list = []

    metric = MeanAveragePrecision()
    model.eval()

    for iter_num, data in enumerate(loader):


        if data['annot'][0,:, 4][0] == 0 or data['annot'][0, :, 4].shape[0] > 1 :
            continue
        
        scores, classification, transformed_anchors = model(data['img'].float())
        if len(scores) == 0:
            continue
        idxs= np.argmax(scores.cpu().detach().numpy())
        
        num_pos += 1
        # print(transformed_anchors.cpu()[idxs].unsqueeze(0))
        # print(data['annot'])
        iou = calc_iou(transformed_anchors.cpu()[idxs].unsqueeze(0), data['annot'][0,:, :4])
        annots.append(data['annot'][0,:, :4].squeeze(0))
        predicted_box.append(transformed_anchors.cpu()[idxs])
        scores_list.append(scores[idxs])

        IOU += iou.item()

        preds = [
            dict(boxes=transformed_anchors.cpu()[idxs].unsqueeze(0), scores=scores[idxs].unsqueeze(0), labels=torch.tensor([1])),
            ]

        target = [
            dict(boxes=data['annot'][0,:, :4], labels=torch.tensor([1])),

        ]
        map_50 += metric(preds,target)['map_50']

    return calculate_recall_precision(torch.stack(annots), torch.stack(predicted_box), torch.stack(scores_list)), IOU/num_pos, (map_50/ num_pos).item()
    
    