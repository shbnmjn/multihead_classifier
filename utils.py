import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path
from sklearn.utils import shuffle
from tensorflow import keras
import cv2
import json


def car_dataset(path):
    color_map = {}
    type_map = {}
    color_idx = 0
    images_path = []
    types_label = []
    colors_label = []
    car_types = os.listdir(path)
    for car_type in tqdm(range(len(car_types))):
        type_map[car_types[car_type]] = car_type
        car_colors = os.listdir(path + car_types[car_type])

        for car_color in car_colors:
            if car_color not in color_map.keys():
                color_map[car_color] = color_idx
                color_idx += 1

            images = os.listdir(path + car_types[car_type] + '/' + car_color)
            for image in images:
                images_path.append(car_types[car_type] + '/' + car_color + '/' + image)
                types_label.append(car_type)
                colors_label.append(color_map[car_color])

    labels = pd.DataFrame({'images': images_path, 'types_label': types_label, 'colors_label': colors_label})
    labels = shuffle(labels)
    length = len(labels)
    train_size = int(length * 0.8)
    labels[:train_size].to_csv('train.csv')
    labels[:train_size].to_csv('val.csv')
    labels_json = {'type': {value: key for key, value in type_map.items()},
                   'color': {value: key for key, value in color_map.items()}}
    with open('dataset/train/train.json', 'w') as json_obj:
        json.dump(labels_json, json_obj, indent=4)


def keras_callbacks():

    callbacks = [
        keras.callbacks.TensorBoard(
            log_dir='logs/',
            histogram_freq=0,
            write_graph=True,
            write_images=True,
            update_freq="epoch",
            profile_batch=0,
            embeddings_freq=0,
            embeddings_metadata=None
        ),
        keras.callbacks.ModelCheckpoint("checkpoint/", save_best_only=True),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.CSVLogger(os.path.join('logs', "result.csv"), separator=",",
                                  append=False)
    ]
    return callbacks



