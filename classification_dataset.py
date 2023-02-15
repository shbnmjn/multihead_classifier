import os
import cv2
import keras.utils
import numpy as np
import pandas as pd


class DataGenerator(keras.utils.Sequence):
    def __init__(self,
                 images_path,
                 label_csv,
                 dim,
                 batch_size=16,
                 channels=3,
                 pad_resize=True,
                 shuffle=True):
        super(DataGenerator, self).__init__()

        self.images_path = images_path
        self.label_csv = label_csv
        self.label1, self.label2, self.images_list = self._list_ids
        self.dim = dim
        self.channels = channels
        self.shuffle = shuffle
        self.pad_resize = pad_resize
        self.batch_size = batch_size
        self.on_epoch_end()

    @property
    def _list_ids(self):
        data = pd.read_csv(self.label_csv)
        images = list(data['images'].values)
        label1 = list(data['types_label'].values)
        label2 = list(data['colors_label'].values)
        
        return label1, label2, images

    def __len__(self):
        return int(np.floor(len(self.images_list) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_list))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _pad_resize(self, image, new_shape=(224, 224), color=(114, 114, 114), scaleup=False):

        shape = image.shape[:2]  # current shape [height, width]

        if isinstance(self.dim, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return image, r, (dw, dh)

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_image_name = [self.images_list[idx] for idx in indexes]
        X = np.empty((self.batch_size, *self.dim, self.channels))
        y1 = np.empty(self.batch_size, dtype=int)
        y2 = np.empty(self.batch_size, dtype=int)

        for i, name in enumerate(batch_image_name):
            image_path = os.path.join(self.images_path, name)
            image = cv2.imread(image_path)
            if self.pad_resize:
                image, r, pad = self._pad_resize(image, self.dim)
            else:
                image = cv2.resize(image, self.dim)
            X[i, ...] = image
            y1[i] = self.label1[indexes[i]]
            y2[i] = self.label2[indexes[i]]
        return X, [y1, y2]
