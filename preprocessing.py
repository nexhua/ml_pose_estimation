"""
Involves operations for the preprocessing operations.
Generates y_train list(Sorted, normalized)
"""

import os
import numpy as np
from annot import annotations
import configurations


class Label:
    def __init__(self):
        self.data_loader = annotations.Data()
        self.config = configurations.Configuration()

    def create_sorted_y_labels(self, file_names, labels):
        print("STARTING Y LABEL CREATION")
        y = np.ndarray(shape=(len(file_names), self.config.DIM))
        print(y.shape)
        for i, name in enumerate(file_names):
            found = next((x for x in labels if x["image"] == name), None)
            assert(found is not None)
            y[i] = np.concatenate(
                [np.array(found["joints"]).ravel(), np.array(found["center"])])
        print("Y LABEL CREATION SUCCESS")
        return y

    def create_sorted_resized_y_label(self, type):
        train_labels = self.data_loader.get_train_data()
        print("TRAIN LABEL FETCH SUCCESSFULL")

        sorted_train_names = sorted(os.listdir(
            os.path.join(self.config.IMAGES_BASE_DIR, type)))
        print("TRAIN IMAGE FILE FETCH SUCCESSFULL")

        return self.create_sorted_y_labels(sorted_train_names, train_labels)
