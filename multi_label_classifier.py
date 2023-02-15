import tensorflow as tf
from tensorflow import keras


def transfer_model():
    model = tf.keras.applications.MobileNet(include_top=True)
    base_input = model.layers[0].input
    out = model.layers[-2].output
    out1 = tf.keras.layers.Dense(100, activation="softmax", name="type")(out)
    out2 = tf.keras.layers.Dense(13, activation="softmax", name="color")(out)
    new_model = keras.Model(inputs=base_input, outputs=[out1, out2])
    return new_model
