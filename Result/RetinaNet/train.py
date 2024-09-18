from config import (
    DEVICE,
    NUM_CLASSES,
    NUM_EPOCHS,
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES,
    NUM_WORKERS,
    RESIZE_TO,
    VALID_DIR,
    TRAIN_DIR,
    CONFIDENCE_THRESHOLD
)
from eval import (
    validate,
    validate_loss
    # validate_and_compute_loss
)
import numpy as np
from sklearn.metrics import precision_score, recall_score
from model import create_model
from custom_utils import (
    Averager,
    SaveBestModel,
    save_model,
    save_mAP,
    save_loss_plot
)
from tqdm.auto import tqdm
from datasets import (
    create_train_dataset,
    create_valid_dataset,
    create_train_loader,
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import time
import torch
plt.style.use('ggplot')
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.detection import retinanet_resnet50_fpn
import os
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


# Function for running training iterations.
# train_loss_results = []
def train(train_data_loader, model):
    print('Training')
    model.train()

    # Initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    total_loss = 0.0
    total_cls_loss = 0.0
    total_bbox_loss = 0.0
    num_batches = 0

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        total_loss += loss_value

        cls_losses = loss_dict['classification']
        total_cls_loss += cls_losses.item()

        bbox_losses = loss_dict['bbox_regression']
        total_bbox_loss += bbox_losses.item()

        num_batches += 1

        losses.backward()
        optimizer.step()

        # Update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    # Compute average losses
    avg_loss = total_loss / num_batches
    avg_cls_loss = total_cls_loss / num_batches
    avg_bbox_loss = total_bbox_loss / num_batches

    # Append the results to the local file
    with open("outputs/train_loss.txt", 'a') as f:
        f.write(f" avg Loss: {avg_loss:.4f}, avg cls Loss: {avg_cls_loss:.4f}, avg bbox Loss: {avg_bbox_loss:.4f}\n")

    return {'training loss': avg_loss, 'cls loss': avg_cls_loss, 'bbox loss': avg_bbox_loss}


if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    train_dataset = create_train_dataset(TRAIN_DIR) # TRAIN_DIR
    valid_dataset = create_valid_dataset(VALID_DIR) # VALID_DIR
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES)

    checkpoint = torch.load('/notebooks/retinaNet/outputs/last_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model's weights
    model = model.to(DEVICE)

    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=15, gamma=0.1, verbose=True
    )

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss, mAP and validation loss values
    train_loss_list = []
    cls_loss_list = []
    bbox_list = []

    map_50_list = []
    map_list = []
    valid_precision_list = []
    valid_recall_list = []
    valid_loss_list = []
    valid_cls_loss_list = []
    valid_bbox_list = []

    # Mame to save the trained model with.
    MODEL_NAME = 'model'

    # Whether to show transformed images from data loader or not.
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image

        show_tranformed_image(train_loader)

    # To save best model.
    save_best_model = SaveBestModel()

    metric = MeanAveragePrecision()
    # Training loop.
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch + 1} of {NUM_EPOCHS}")

        # Reset the training loss histories for the current epoch.
        train_loss_hist.reset()

        # Start timer and carry out training and validation.
        start = time.time()
        train_loss = train(train_loader, model)
        metric = validate_loss(valid_loader, model)
        metric_summary = validate(valid_loader, model,DEVICE)

        print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        print(f"Epoch #{epoch + 1} mAP: {metric_summary['map']}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

        train_loss_list.append(train_loss['training loss'])
        cls_loss_list.append(train_loss['cls loss'])
        bbox_list.append(train_loss['bbox loss'])

        map_50_list.append(metric_summary['map_50'])
        map_list.append(metric_summary['map'])
        valid_precision_list.append(metric_summary['precision'])
        valid_recall_list.append(metric_summary['recall'])
        valid_loss_list.append(metric['validate loss'])
        valid_cls_loss_list.append(metric['valid cls loss'])
        valid_bbox_list.append(metric['valid bbox loss'])
        # save the best model till now.
        save_best_model(
            model, float(metric_summary['map']), epoch, 'outputs'
        )
        # Save the current epoch model.
        save_model(epoch, model, optimizer)

        # Save training plot.
        # save_loss_plot(OUT_DIR, train_loss_list, 'training_loss', 'train/loss')
        # save_loss_plot(OUT_DIR, cls_loss_list, 'training_cls_loss', 'train/cls_loss')
        # save_loss_plot(OUT_DIR, bbox_list, 'training_bb_loss', 'train/bbox_loss')
        # # Save validating plot.
        # save_mAP(OUT_DIR, map_50_list, map_list)
        # save_loss_plot(OUT_DIR, valid_precision_list, 'validating_precision', 'valid/precision')
        # save_loss_plot(OUT_DIR, valid_loss_list, 'validating_loss', 'valid/loss')
        # save_loss_plot(OUT_DIR, valid_cls_loss_list, 'validating_cls_loss', 'valid/cls_loss')
        # save_loss_plot(OUT_DIR, valid_bbox_list, 'validating_bb_loss', 'valid/bbox_loss')

        # scheduler.step()