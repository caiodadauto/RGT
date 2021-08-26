import os
from time import time
from datetime import datetime
from functools import partial

import numpy as np
import sonnet as snt
import mlflow as mlf
import tensorflow as tf
from tqdm import tqdm
from hydra.utils import get_original_cwd
from graph_nets.utils_tf import specs_from_graphs_tuple
from gn_contrib.train import binary_crossentropy

from rgt.gn_modules import RoutingGraphTransformer
from rgt.utils import init_generator, get_bacc, get_f1, get_precision
from rgt.utils_mlf import save_pickle, load_pickle, log_params_from_omegaconf_dict, set_mlflow

__all__ = ["EstimatorRGT", "Train"]


class Train:
    def __init__(
        self,
        exp_name,
        exp_tags,
        run_tags,
        run_id,
        get_last_run,
        cfg_model,
        cfg_estimator,
    ):
        self._run = set_mlflow(
            exp_name, exp_tags, run_tags, run_id, get_last_run
        )
        start_epoch = self._run.data.metrics.get("epoch")
        seen_graphs = self._run.data.metrics.get("graph")
        start_epoch = int(start_epoch) if start_epoch is not None else 0
        seen_graphs = int(seen_graphs) if seen_graphs is not None else 0
        with mlf.start_run(run_id=self._run.info.run_id):
            log_params_from_omegaconf_dict(cfg_model)
            log_params_from_omegaconf_dict(cfg_estimator)
            model = RoutingGraphTransformer(**cfg_model)
            estimator = EstimatorRGT(model, start_epoch, seen_graphs, **cfg_estimator)
            estimator.train()


class EstimatorRGT(snt.Module):
    def __init__(
        self,
        rgt,
        start_epoch,
        seen_graphs,
        num_epochs,
        optimizer,
        init_lr,
        decay_steps,
        end_lr,
        power,
        cycle,
        tr_size,
        tr_batch_size,
        val_batch_size,
        tr_path_data,
        val_path_data,
        file_ext,
        seed,
        msg_ratio=0.45,
        node_fields=None,
        edge_fields=None,
        class_weights=[1.0, 1.0],
        scaler=True,
        delta_time_validation=60,
        compile=False,
    ):
        super(EstimatorRGT, self).__init__(name="EstimatorRGT")
        self._seed = seed
        self._start_epoch = start_epoch
        self._seen_graphs = seen_graphs
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)
        try:
            self._rs = load_pickle("random_state")
        except Exception:
            self._rs = np.random.RandomState(self._seed)

        self._best_acc = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._best_f1 = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._best_precision = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._best_delta = tf.Variable(np.infty, trainable=False, dtype=tf.float32)
        self._delta_time_validation = delta_time_validation

        self._model = rgt
        self._tr_size = tr_size
        self._num_epochs = num_epochs
        self._tr_batch_size = tr_batch_size
        self._val_batch_size = val_batch_size
        self._loss_fn = partial(
            binary_crossentropy,
            entity="edges",
            class_weights=tf.constant(class_weights, dtype=tf.float32),
            min_num_msg=tf.cast(
                tf.math.ceil(msg_ratio * self._model._num_msg), dtype=tf.int32
            ),
        )

        self._lr = tf.Variable(init_lr, trainable=False, dtype=tf.float32, name="lr")
        self._step = tf.Variable(0, trainable=False, dtype=tf.float32, name="tr_step")
        self._opt = snt.optimizers.__getattribute__(optimizer)(learning_rate=self._lr)
        self._schedule_lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(
            init_lr, decay_steps, end_lr, power=power, cycle=cycle
        )

        if node_fields is None and edge_fields is None:
            input_fields = None
        else:
            input_fields = dict()
            if node_fields is not None:
                input_fields["node"] = node_fields
            if edge_fields is not None:
                input_fields["edge"] = edge_fields
        self._input_fields = input_fields
        self._tr_path_data = tr_path_data
        self._val_path_data = val_path_data
        self._batch_generator = partial(
            init_generator,
            scaler=scaler,
            random_state=self._rs,
            file_ext=file_ext,
            input_fields=input_fields,
        )
        self.set_managers()

        if compile:
            val_generator = self._batch_generator(
                self._val_path_data, self._val_batch_size
            )
            in_val_graphs, gt_val_graphs, _ = next(val_generator)
            in_signature = specs_from_graphs_tuple(in_val_graphs, True)
            gt_signature = specs_from_graphs_tuple(gt_val_graphs, True)
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

    def set_managers(self):
        artifact_path = mlf.get_artifact_uri()
        last_path = os.path.join(artifact_path, "last_ckpts")
        acc_path = os.path.join(artifact_path, "best_acc_ckpts")
        precision_path = os.path.join(artifact_path, "best_precision_ckpts")
        f1_path = os.path.join(artifact_path, "best_f1_ckpts")
        delta_path = os.path.join(artifact_path, "best_delta_ckpts")

        test = mlf.models.Model(last_path)
        print()
        print(test)
        print()

        ckpt = tf.train.Checkpoint(
            step=self._step,
            optimizer=self._opt,
            model=self._model,
            best_acc=self._best_acc,
            best_f1=self._best_f1,
            best_precision=self._best_precision,
            best_delta=self._best_delta,
        )
        self._last_ckpt_manager = tf.train.CheckpointManager(
            ckpt, last_path, max_to_keep=1
        )
        self._best_acc_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            acc_path,
            max_to_keep=3,
        )
        self._best_f1_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            f1_path,
            max_to_keep=3,
        )
        self._best_precision_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            precision_path,
            max_to_keep=3,
        )
        self._best_delta_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            delta_path,
            max_to_keep=3,
        )
        if os.path.isdir(os.path.normpath(last_path).split(":")[-1]):
            _ = ckpt.restore(self._last_ckpt_manager.latest_checkpoint)
            print(
                "Restore training from {}\n\tEpoch : {}, "
                "Seen Graphs : {}, Best Acc : {}, Best F1 : {}"
                "Best Precision : {}, Best Delta : {}\n".format(
                    last_path,
                    self._best_acc.numpy(),
                    self._best_f1.numpy(),
                    self._best_precision.numpy(),
                    self._best_delta.numpy(),
                    self._start_epoch,
                    self._seen_graphs,
                )
            )

    def __update_model_weights(self, in_graphs, gt_graphs):
        targets = in_graphs.globals
        with tf.GradientTape() as tape:
            output_graphs = self._model(in_graphs, targets, True)
            loss = self._loss_fn(gt_graphs.edges, output_graphs)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._opt.apply(gradients, self._model.trainable_variables)
        self._step.assign_add(1)
        self._lr.assign(self._schedule_lr_fn(self._step))
        return output_graphs[-1], loss

    def __eval(self, in_graphs):
        targets = in_graphs.globals
        output_graphs = self._model(in_graphs, targets, False)
        return output_graphs

    def __assess_val(self):
        acc = []
        f1 = []
        precision = []
        loss = []
        val_generator = self._batch_generator(self._val_path_data, self._val_batch_size)
        for in_graphs, gt_graphs, _ in val_generator:
            out_graphs = self._eval(in_graphs)
            acc.append(get_bacc(gt_graphs.edges, out_graphs[-1].edges))
            f1.append(get_f1(gt_graphs.edges, out_graphs[-1].edges))
            precision.append(get_precision(gt_graphs.edges, out_graphs[-1].edges))
            loss.append(self._loss_fn(gt_graphs.edges, out_graphs))
        return (
            tf.reduce_sum(acc) / len(acc),
            tf.reduce_sum(f1) / len(f1),
            tf.reduce_sum(precision) / len(precision),
            tf.reduce_sum(loss) / len(loss),
        )

    def __log_scalars(self, params):
        with self._writer_scalars.as_default():
            for name, value in params.items():
                tf.summary.scalar(name, data=value, step=tf.cast(self._step, tf.int64))
            self._writer_scalars.flush()

    def train(self):
        start_time = time()
        last_validation = start_time
        for epoch in range(self._start_epoch, self._num_epochs):
            epoch_bar = tqdm(
                total=self._tr_size,
                initial=self._seen_graphs,
                desc="Processed Graphs",
                leave=False,
            )
            epoch_bar.set_postfix(epoch="{} / {}".format(epoch, self._num_epochs))
            tr_generator = self._batch_generator(
                self._tr_path_data,
                self._tr_batch_size,
                size=self._tr_size,
                seen_graphs=self._seen_graphs,
            )
            for in_graphs, gt_graphs, _ in tr_generator:
                tr_out_graphs, tr_loss = self._update_model_weights(
                    in_graphs, gt_graphs
                )
                delta_time = time() - last_validation
                if delta_time >= self._delta_time_validation:
                    last_validation = time()
                    tr_acc = get_bacc(gt_graphs.edges, tr_out_graphs.edges)
                    tr_f1 = get_f1(gt_graphs.edges, tr_out_graphs.edges)
                    tr_precision = get_precision(gt_graphs.edges, tr_out_graphs.edges)
                    val_acc, val_f1, val_precision, val_loss = self.__assess_val()
                    delta = tf.abs(tr_loss - val_loss)

                    save_pickle("random_state", self._rs)
                    mlf.log_metric("graph", self._seen_graphs)
                    mlf.log_metric("epoch", epoch)
                    mlf.log_metrics(
                        {
                            "tr loss": tr_loss.numpy(),
                            "tr f1": tr_f1.numpy(),
                            "tr bacc": tr_acc.numpy(),
                            "tr precision": tr_precision.numpy(),
                            "val loss": val_loss.numpy(),
                            "val f1": val_f1.numpy(),
                            "val bacc": val_acc.numpy(),
                            "val precision": val_precision.numpy(),
                            "delta": delta.numpy(),
                            "lr": self._lr.numpy(),
                        }
                    )
                    self._last_ckpt_manager.save()
                    if self._best_acc <= val_acc:
                        self._best_acc_ckpt_manager.save()
                        self._best_acc.assign(val_acc)
                    if self._best_f1 <= val_f1:
                        self._best_f1_ckpt_manager.save()
                        self._best_f1.assign(val_f1)
                    if self._best_precision <= val_precision:
                        self._best_precision_ckpt_manager.save()
                        self._best_precision.assign(val_precision)
                    if self._best_delta >= delta:
                        self._best_delta_ckpt_manager.save()
                        self._best_delta.assign(delta)
                    epoch_bar.set_postfix(
                        epoch="{} / {}".format(epoch, self._num_epochs),
                        tr_loss="{:.4f}".format(tr_loss.numpy()),
                        val_loss="{:.4f}".format(val_loss.numpy()),
                        best_acc="{:.4f}".format(self._best_acc.numpy()),
                        best_f1="{:.4f}".format(self._best_f1.numpy()),
                        best_precision="{:.4f}".format(self._best_precision.numpy()),
                        best_delta="{:.4f}".format(self._best_delta.numpy()),
                    )
                self._seen_graphs += in_graphs.n_node.shape[0]
                epoch_bar.update(in_graphs.n_node.shape[0])
            epoch_bar.close()
            self._seen_graphs = 0
