import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score

import pytop
from gn_contrib.utils import networkx_to_graph_tuple_generator


__all__ = ["init_generator", "get_bacc", "get_f1", "get_precision"]


def init_generator(
    path,
    n_batch,
    scaler,
    random_state,
    file_ext,
    seen_graphs=0,
    size=None,
    input_fields=None,
):

    if scaler:
        _scaler = minmax_scale
    else:
        _scaler = None
    generator = networkx_to_graph_tuple_generator(
        pytop.batch_files_generator(
            path,
            file_ext,
            n_batch,
            dataset_size=size,
            shuffle=True,
            bidim_solution=False,
            input_fields=input_fields,
            random_state=random_state,
            seen_graphs=seen_graphs,
            scaler=_scaler,
        )
    )
    return generator


def get_bacc(expected, predicted, th=0.5):
    e = expected.numpy()
    p = (predicted.numpy() >= th).astype(np.int32)
    return tf.constant(balanced_accuracy_score(e, p), dtype=tf.float32)


def get_precision(expected, predicted, th=0.5):
    e = expected.numpy()
    p = (predicted.numpy() >= th).astype(np.int32)
    return tf.constant(precision_score(e, p), dtype=tf.float32)


def get_f1(expected, predicted, th=0.5):
    e = expected.numpy()
    p = (predicted.numpy() >= th).astype(np.int32)
    return tf.constant(f1_score(e, p), dtype=tf.float32)
