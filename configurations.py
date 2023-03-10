"""
Config file for the entire Machine Learning Project
"""


class Configuration:
    def __init__(self):
        self.IMAGES_BASE_DIR = "images"
        self.ANNOTATIONS_BASE_DIR = "annot"
        self.ORIGINAL_DATASET_BASE_DIR = "images\\all\\images"
        self.KEYPOINT_COUNT = 16
        self.DIM = self.KEYPOINT_COUNT * 2 + 2
        self.INPUT_DIMS = (224, 224, 3)
        self.BATCH_SIZE = 32
        self.EPOCHS = 10
        self.LEN_TRAIN_IMGS = 14679
        self.LEN_TRAIN_SPLIT = 11744
        self.LEN_VALIDATION_SPLIT = 2935
        self.VALIDATION_RATIO = 0.2
