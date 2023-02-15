import config as cfg
import tensorflow as tf
from utils import keras_callbacks
from tensorflow import keras
from classification_dataset import DataGenerator
from multi_label_classifier import transfer_model

train_dataset = DataGenerator(
    images_path=cfg.dataset_path,
    label_csv=cfg.train_labels,
    dim=cfg.train_image_size,
    pad_resize=cfg.pad_resize,
    batch_size=cfg.batch_size)

val_dataset = DataGenerator(
    images_path=cfg.dataset_path,
    label_csv=cfg.val_labels,
    dim=cfg.train_image_size,
    pad_resize=cfg.pad_resize,
    batch_size=cfg.batch_size)

model = transfer_model()
print(model.summary())
if cfg.model_plot:
    tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

if __name__ == "__main__":

    model.fit(train_dataset,
              epochs=cfg.epoch_num,
              validation_data=val_dataset,
              callbacks=keras_callbacks())
