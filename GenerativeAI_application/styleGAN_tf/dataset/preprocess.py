from functools import partial

import tensorflow as tf
import numpy as np

import keras

def log2(x):
    return int(np.log2(x))

def resize_image(res, image):
    # only downsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image

def create_dataloader(res):
    batch_size = batch_sizes[log2(res)]
    # NOTE: we unbatch the dataset so we can `batch()` it again with the `drop_remainder=True` option
    # since the model only supports a single batch size
    dl = ds_train.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE).unbatch()
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl

batch_sizes = {2: 16, 3: 16, 4: 16, 5: 16, 6: 16, 7: 8, 8: 4, 9: 2, 10: 1}
train_step_ratio = {k: batch_sizes[2] / v for k, v in batch_sizes.items()}

ds_train = keras.utils.image_dataset_from_directory(
    "celeba_gan", label_mode=None, image_size=(64, 64), batch_size=32
)
