from typing import Mapping, Dict
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds

def _preprocess_image_dataset(
    element: Mapping[str, tf.Tensor],
    num_labels: int
) -> Dict[str, tf.Tensor]:
    rescaled_image = tf.cast(element["image"], tf.float32) / 255.
    one_hot_label = tf.one_hot(tf.cast(element["label"], tf.int32), num_labels)
    
    return {"image": rescaled_image, "label": one_hot_label}

def load_mnist(
    batch_size: int,
    preprocess: bool = True,
    split: tfds.Split = tfds.Split.TRAIN,
    shuffle: bool = True,
    buffer_size: int = 10_000
) -> tf.data.Dataset:
    
    dataset = tfds.load("mnist", split=split)
    if shuffle:
        dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    if preprocess:
        dataset = (
            dataset.map(
                partial(_preprocess_image_dataset, num_labels=10),
                num_parallel_calls=tf.data.AUTOTUNE,
            )
            .prefetch(tf.data.AUTOTUNE)
        )
    
    return dataset

