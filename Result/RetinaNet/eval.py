import torch

from tqdm import tqdm
from config import DEVICE, NUM_CLASSES, NUM_WORKERS
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from model import create_model
from datasets import create_valid_dataset, create_valid_loader
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix

from torchvision.ops import box_iou

from torchvision.ops import box_iou

# def validate(valid_data_loader, model):
#     model.eval()
#     print('Validating')

#     # Initialize tqdm progress bar.
#     prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
#     target = []
#     preds = []
#     precision_scores = []
#     recall_scores = []

#     for batch_idx, data in enumerate(prog_bar):
#         images, targets = data

#         images = list(image.to(DEVICE) for image in images)
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#         with torch.no_grad():
#             outputs = model(images, targets)

#         # For mAP calculation using Torchmetrics.
#         #####################################
#         for i in range(len(images)):
#             true_dict = dict()
#             preds_dict = dict()
#             true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
#             true_dict['labels'] = targets[i]['labels'].detach().cpu()
#             preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
#             preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
#             preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
#             preds.append(preds_dict)
#             target.append(true_dict)
#         #####################################
#             true_boxes_i = targets[i]['boxes'].cpu().numpy()
#             pred_boxes_i = outputs[i]['boxes'].cpu().numpy()

#             iou = box_iou(torch.tensor(true_boxes_i), torch.tensor(pred_boxes_i))

#             # Define a threshold for IoU
#             iou_threshold = 0.5

#             # True positives: IoU > threshold
#             tp = (iou > iou_threshold).sum().item()

#             # False positives: predicted boxes which have IoU < threshold
#             fp = (iou <= iou_threshold).sum().item()

#             # False negatives: ground truth boxes which have IoU < threshold
#             fn = len(true_boxes_i) - tp

#             # Calculate precision and recall
#             precision_i = tp / (tp + fp) if tp + fp > 0 else 0
#             recall_i = tp / (tp + fn) if tp + fn > 0 else 0

#             precision_scores.append(precision_i)
#             recall_scores.append(recall_i)

#     metric = MeanAveragePrecision()
#     metric.update(preds, target)
#     metric_summary = metric.compute()
#     map_50 = metric_summary['map_50']
#     map = metric_summary['map']

#     avg_precision = sum(precision_scores) / len(precision_scores)
#     avg_recall = sum(recall_scores) / len(recall_scores)

#     with open("outputs/valid_mAP.txt", 'a') as f:
#         f.write(f" map_50: {map_50:.4f}, map: {map:.4f}, precision: {avg_precision:.4f}, recall: {avg_recall:.4f}\n")

#     return {'map_50': map_50, 'map': map, 'precision': avg_precision, 'recall': avg_recall}
from collections import defaultdict


def validate(valid_data_loader, model, DEVICE):
    model.eval()
    print('Validating')

    # Initialize tqdm progress bar.
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    precision_scores = defaultdict(int)
    recall_scores = defaultdict(int)
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for batch_idx, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            outputs = model(images, targets)

        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)

            true_boxes_i = targets[i]['boxes'].cpu().numpy()
            pred_boxes_i = outputs[i]['boxes'].cpu().numpy()

            iou = box_iou(torch.tensor(true_boxes_i), torch.tensor(pred_boxes_i))

            # Define a threshold for IoU
            iou_threshold = 0.5

            # True positives: IoU > threshold
            tp_i = (iou > iou_threshold).sum().item()

            # False positives: predicted boxes which have IoU < threshold
            fp_i = (iou <= iou_threshold).sum().item()

            # False negatives: ground truth boxes which have IoU < threshold
            fn_i = len(true_boxes_i) - tp_i

            # Calculate precision and recall
            precision_i = tp_i / (tp_i + fp_i) if tp_i + fp_i > 0 else 0
            recall_i = tp_i / (tp_i + fn_i) if tp_i + fn_i > 0 else 0

            for label in targets[i]['labels'].cpu().numpy():
                tp[label] += tp_i
                fp[label] += fp_i
                fn[label] += fn_i
                precision_scores[label] += precision_i
                recall_scores[label] += recall_i

    # Calculate average precision and recall for each class
    for label in precision_scores.keys():
        precision_scores[label] /= len(valid_data_loader)
        recall_scores[label] /= len(valid_data_loader)

    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    map_50 = metric_summary['map_50']
    map = metric_summary['map']

    with open("outputs/valid_mAP.txt", 'a') as f:
        f.write(f" map_50: {map_50:.4f}, map: {map:.4f}\n")
        for label, precision in precision_scores.items():
            f.write(f"Class {label} precision: {precision:.4f}\n")

    return {'map_50': map_50, 'map': map, 'precision': precision_scores, 'recall': recall_scores}
# # Evaluation function
# # def validate(valid_data_loader, model):
# #     model.eval()
# #     print('Validating')

# #     # Initialize tqdm progress bar.
# #     prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
# #     target = []
# #     preds = []
# #     precision_scores = []
# #     recall_scores = []

# #     for batch_idx, data in enumerate(prog_bar):
# #         images, targets = data

# #         images = list(image.to(DEVICE) for image in images)
# #         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

# #         with torch.no_grad():
# #             outputs = model(images, targets)

# #         # For mAP calculation using Torchmetrics.
# #         #####################################
# #         for i in range(len(images)):
# #             true_dict = dict()
# #             preds_dict = dict()
# #             true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
# #             true_dict['labels'] = targets[i]['labels'].detach().cpu()
# #             preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
# #             preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
# #             preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
# #             preds.append(preds_dict)
# #             target.append(true_dict)
# #         #####################################
# #             true_labels_i = targets[i]['labels'].cpu().numpy()
# #             pred_labels_i = outputs[i]['labels'].cpu().numpy()

# #             precision_i = 0
# #             recall_i = 0

# #             if len(true_labels_i) == len(pred_labels_i):
# #                 precision_i = precision_score(true_labels_i, pred_labels_i, average='weighted', zero_division=0)
# #                 recall_i = recall_score(true_labels_i, pred_labels_i, average='weighted', zero_division=0)
# #             precision_scores.append(precision_i)
# #             recall_scores.append(recall_i)

# #     metric = MeanAveragePrecision()
# #     metric.update(preds, target)
# #     metric_summary = metric.compute()
# #     map_50 = metric_summary['map_50']
# #     map = metric_summary['map']
# #     # precision = Precision(task="multiclass", average='macro', num_classes=8)
# #     # precision(preds, target)
# #     # recall = Recall(task="multiclass", average='macro', num_classes=8)
# #     # recall(preds, target)
# #     #avg_precision = 0
# #     #avg_recall =  0
# #     avg_precision = sum(precision_scores) / len(precision_scores)
# #     avg_recall = sum(recall_scores) / len(recall_scores)

# #     with open("outputs/valid_mAP.txt", 'a') as f:
# #         f.write(f" map_50: {map_50:.4f}, map: {map:.4f}, precision: {avg_precision:.4f}, recall: {avg_recall:.4f}\n")

# #     return {'map_50': map_50, 'map': map, 'precision': avg_precision, 'recall': avg_recall}


def validate_loss(valid_data_loader, model):
    print('Validating')
    model.train()

    # Initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    num_batches = 0

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        total_loss += losses.item()

        cls_losses = loss_dict['classification']
        total_cls_loss += cls_losses.item()

        bbox_losses = loss_dict['bbox_regression']
        total_bbox_loss += bbox_losses.item()

        num_batches += 1

    # Compute average losses
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_bbox_loss = total_bbox_loss / num_batches
    # Append the results to the local file
    with open("outputs/valid_loss.txt", 'a') as f:
        f.write(f" avg Loss: {avg_loss:.4f}, avg cls Loss: {avg_cls_loss:.4f}, avg bbox Loss: {avg_bbox_loss:.4f}\n")

    return {'validate loss': avg_loss, 'valid cls loss': avg_cls_loss, 'valid bbox loss': avg_bbox_loss}


# from collections import defaultdict
# import numpy as np
# from tqdm import tqdm

# def validate_and_compute_loss(valid_data_loader, model, DEVICE):
#     print('Validating')
#     model.eval()

#     prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
#     target = []
#     preds = []
#     total_loss = 0.0
#     total_cls_loss = 0.0
#     total_bbox_loss = 0.0
#     num_batches = 0

#     tp_dict = defaultdict(int)
#     fp_dict = defaultdict(int)
#     fn_dict = defaultdict(int)

#     for batch_idx, data in enumerate(prog_bar):
#         images, targets = data

#         images = [image.to(DEVICE) for image in images]
#         targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

#         with torch.no_grad():
#             outputs = model(images)

#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
#         total_loss += losses.item()
#         cls_losses = loss_dict['classification']
#         total_cls_loss += cls_losses.item()
#         bbox_losses = loss_dict['bbox_regression']
#         total_bbox_loss += bbox_losses.item()
#         num_batches += 1

#         for i in range(len(images)):
#             true_dict = dict()
#             preds_dict = dict()
#             true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
#             true_dict['labels'] = targets[i]['labels'].detach().cpu()
#             preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
#             preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
#             preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
#             preds.append(preds_dict)
#             target.append(true_dict)

#             true_boxes_i = targets[i]['boxes'].cpu().numpy()
#             pred_boxes_i = outputs[i]['boxes'].cpu().numpy()

#             iou = box_iou(torch.tensor(true_boxes_i), torch.tensor(pred_boxes_i))

#             iou_threshold = 0.5

#             true_labels_i = targets[i]['labels'].cpu().numpy()
#             pred_labels_i = outputs[i]['labels'].cpu().numpy()

#             for label in np.unique(np.concatenate([true_labels_i, pred_labels_i])):
#                 tp = ((pred_labels_i == label) & (iou > iou_threshold)).sum().item()
#                 fp = ((pred_labels_i == label) & (iou <= iou_threshold)).sum().item()
#                 fn = ((true_labels_i == label) & (iou <= iou_threshold)).sum().item()

#                 tp_dict[label] += tp
#                 fp_dict[label] += fp
#                 fn_dict[label] += fn

#     metric = MeanAveragePrecision()
#     metric.update(preds, target)
#     metric_summary = metric.compute()
#     map_50 = metric_summary['map_50']
#     map = metric_summary['map']

#     precision_dict = {label: tp / (tp + fp) if tp + fp > 0 else 0 for label, tp, fp in zip(tp_dict.keys(), tp_dict.values(), fp_dict.values())}

#     with open("outputs/class_precisions.txt", 'a') as f:
#         for label, precision in precision_dict.items():
#             f.write(f"Class {label}: Precision {precision:.4f}\n")

#     avg_loss = total_loss / num_batches
#     avg_cls_loss = total_cls_loss / num_batches
#     avg_bbox_loss = total_bbox_loss / num_batches

#     with open("outputs/valid_mAP.txt", 'a') as f:
#         f.write(f" map_50: {map_50:.4f}, map: {map:.4f}, avg Loss: {avg_loss:.4f}, avg cls Loss: {avg_cls_loss:.4f}, avg bbox Loss: {avg_bbox_loss:.4f}\n")

#     return {'map_50': map_50, 'map': map, 'class_precisions': precision_dict, 'validate loss': avg_loss, 'valid cls loss': avg_cls_loss, 'valid bbox loss': avg_bbox_loss}

if __name__ == '__main__':
    # Load the best model and trained weights.
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('outputs/last_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    test_dataset = create_valid_dataset(
        'data/BCCD/test'
    )
    test_loader = create_valid_loader(test_dataset, num_workers=NUM_WORKERS)
    metric_summary = validate(test_loader, model,DEVICE)
    print(metric_summary)
    metrics = validate_loss(test_loader, model)
    print(metrics)
    # metric_summary = validate_and_compute_loss(test_loader, model)
    # print(metric_summary)


