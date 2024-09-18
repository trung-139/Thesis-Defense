import torch


BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 100 # Number of epochs to train for.
NUM_WORKERS = 8 # Number of parallel workers for data loading.
CONFIDENCE_THRESHOLD = 0.5
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#DEVICE = torch.device('cpu')
# Training images and XML files directory.
TRAIN_DIR = '/notebooks/train'
# Validation images and XML files directory.
VALID_DIR = '/notebooks/train'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'no entry', 'no stopping/parking', 'no turning','speed limit','other prohibition signs','warning signs','mandatory signs'
]
NUM_CLASSES = len(CLASSES)
# Whether to visualize images after creating the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False
# Location to save model and plots.
OUT_DIR = '/notebooks/outputs'