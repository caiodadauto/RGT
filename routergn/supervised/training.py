import os
from time import time
from datetime import datetime

import numpy as np
import sonnet as snt
import tensorflow as tf
from tqdm.auto import trange
from tqdm.contrib.itertools import product
from tensorflow.math import log, reduce_mean
from sklearn.preprocessing import minmax_scale
from sklearn.metrics import balanced_accuracy_score
from graph_nets.utils_tf import specs_from_graphs_tuple

from pytop.generator import batch_files_generator
from routergn.nn.utils import networkx_to_graph_tuple_generator


__all__ = ["RGTOptimizer"]


def binary_cross_entropy(expected, predicted, class_weights):
    epsilon = 1e-7
    losses = -1 * (
        expected * log(predicted + epsilon)
        + (1 - expected) * log(1 - predicted + epsilon)
    )
    losses = tf.gather(class_weights, tf.cast(expected, tf.int32)) * losses
    return reduce_mean(losses)


def get_generator(path, batch_size, bidim_solution=False, edge_scaler=minmax_scale):
    return networkx_to_graph_tuple_generator(
        batch_files_generator(
            path, batch_size, bidim_solution=bidim_solution, edge_scaler=edge_scaler
        )
    )


class RGTOptimizer(snt.Module):
    def __init__(
        self,
        rgt,
        optimizer,
        batch_size,
        num_of_epochs,
        path_to_train_data,
        path_to_validation_data,
        final_lr,
        initial_lr,
        delta_steps,
        decay_lr_start_step,
        seed,
        class_weights=[1.0, 1.0],
        loss_fn=binary_cross_entropy,
        edge_scaler=minmax_scale,
        delta_time_to_validate=30,
        log_path="logs/scalars/",
        with_op_graph=True,
        compile=False,
        root_path="",
    ):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        super(RGTOptimizer, self).__init__(name="RGTOptimizer")
        self._model = rgt
        self._best_val_acc = 0
        self._loss_fn = loss_fn
        self._batch_size = batch_size
        self._edge_scaler = edge_scaler
        self._num_of_epochs = num_of_epochs
        self._class_weights = tf.constant(class_weights, dtype=tf.float32)
        self._path_to_train_data = path_to_train_data
        self._path_to_validation_data = path_to_validation_data
        self._delta_time_to_validate = delta_time_to_validate
        self._lr = tf.Variable(
            initial_lr,
            trainable=False,
            dtype=tf.float32,
            name="learning_rate",
        )
        self._opt = optimizer(learning_rate=self._lr)
        self._delta_lr = tf.constant(final_lr - initial_lr, dtype=tf.float32)
        self._initial_lr = tf.constant(initial_lr, dtype=tf.float32, name="initial_lr")
        self._delta_steps = tf.constant(
            delta_steps, dtype=tf.float32, name="delta_steps"
        )
        self._decay_lr_start_step = tf.constant(
            decay_lr_start_step, dtype=tf.float32, name="decay_lr_start_step"
        )
        self._step = tf.Variable(
            0, trainable=False, dtype=tf.float32, name="train_step"
        )
        logdir = os.path.join(
            root_path, log_path, datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        self._writer_scalars = tf.summary.create_file_writer(
            os.path.join(logdir + "/metrics")
        )
        ckpt = tf.train.Checkpoint(
            step=self._step, optimizer=self._opt, net=self._model
        )
        self._last_ckpt_manager = tf.train.CheckpointManager(
            ckpt, os.path.join(logdir, "last_ckpts"), max_to_keep=5
        )
        self._best_ckpt_manager = tf.train.CheckpointManager(
            ckpt, os.path.join(logdir, "best_ckpts"), max_to_keep=5
        )
        val_generator = get_generator(
            self._path_to_validation_data, -1, edge_scaler=self._edge_scaler
        )
        self._in_val_graphs, self._gt_val_graphs, self._raw_edge_val_features = next(
            val_generator
        )
        if compile:
            in_signature = specs_from_graphs_tuple(self._in_val_graphs, True)
            gt_signature = specs_from_graphs_tuple(self._gt_val_graphs, True)
            self._update_model_weights = tf.function(
                self.__update_model_weights,
                input_signature=[in_signature, gt_signature],
            )
            self._eval = tf.function(
                self.__eval,
                input_signature=[in_signature],
            )
        else:
            self._update_model_weights = self.__update_model_weights
            self._eval = self.__eval

    def _update_lr(self):
        if self._step > self._decay_lr_start_step:
            delta_steps = self._delta_steps - self._step
            decay_ratio = 1 - delta_steps / self._delta_steps
            self._lr.assign(self._initial_lr + self._delta_lr * decay_ratio)

    def __update_model_weights(self, in_graphs, gt_graphs):
        loss_for_all_mps = []
        expected = gt_graphs.edges
        targets = in_graphs.globals
        with tf.GradientTape() as tape:
            output_graphs = self._model(in_graphs, targets, True)
            for predicted_graphs in output_graphs:
                predicted = predicted_graphs.edges
                loss_for_all_mps.append(
                    self._loss_fn(expected, predicted, self._class_weights)
                )
            loss = tf.math.reduce_sum(tf.stack(loss_for_all_mps))
            loss = loss / len(output_graphs)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._opt.apply(gradients, self._model.trainable_variables)
        self._step.assign_add(1.0)
        self._update_lr()
        return output_graphs[-1], loss

    def __eval(self, in_graphs):
        targets = in_graphs.globals
        output_graphs = self._model(in_graphs, targets, False)
        return output_graphs[-1]

    def log_scalars(self, params):
        with self._writer_scalars.as_default():
            for name, value in params.items():
                tf.summary.scalar(name, data=value, step=tf.cast(self._step, tf.int64))

    def _get_accuracy(self, predicted, expected, th=0.32):
        float_p = predicted.numpy()
        e = expected.numpy()
        p = (float_p > th).astype(np.int32)
        return balanced_accuracy_score(e, p)

    def train(self):
        start_time = time()
        last_validation = start_time
        for epoch in trange(self._num_of_epochs, desc="Epochs"):
            train_generator = get_generator(
                self._path_to_train_data,
                self._batch_size,
                edge_scaler=self._edge_scaler,
            )
            for ((in_graphs, gt_graphs, raw_edge_features),) in product(
                train_generator, desc="Batchs", leave=False
            ):
                tr_graphs, loss = self._update_model_weights(in_graphs, gt_graphs)
                self.log_scalars({"loss": loss, "learning rate": self._lr})
                delta_time = time() - last_validation
                if delta_time >= self._delta_time_to_validate:
                    val_graphs = self._eval(self._in_val_graphs)
                    last_validation = time()
                    tr_acc = self._get_accuracy(tr_graphs.edges, gt_graphs.edges)
                    val_acc = self._get_accuracy(
                        val_graphs.edges, self._gt_val_graphs.edges
                    )
                    self.log_scalars(
                        {"train accuracy": tr_acc, "val accuracy": val_acc}
                    )
                    self._last_ckpt_manager.save()
                    if self._best_val_acc <= val_acc:
                        self._best_ckpt_manager.save()
