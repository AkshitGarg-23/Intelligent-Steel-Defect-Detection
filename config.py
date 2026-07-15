import os

# Base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_RAW_DIR = os.path.join(BASE_DIR, "dataset", "NEU-DET")
DATASET_SPLIT_DIR = os.path.join(BASE_DIR, "dataset", "split")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Class configurations
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
NUM_CLASSES = len(CLASSES)

# Dataset split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Image configurations
IMAGE_SIZE = (224, 224)

# Normalization parameters (standard ImageNet)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Augmentation hyperparameters
ROTATE_DEGREES = 15
FLIP_PROB = 0.5
BRIGHTNESS = 0.2
CONTRAST = 0.2

# Training hyperparameters
BATCH_SIZE = 32
EPOCHS = 8
LEARNING_RATE_CUSTOM = 0.001
LEARNING_RATE_RESNET = 0.0001
LEARNING_RATE_EFFICIENTNET = 0.0001
WEIGHT_DECAY = 0.0001

# Ensure directories exist
for directory in [CHECKPOINT_DIR, OUTPUT_DIR]:
    os.makedirs(directory, exist_ok=True)
