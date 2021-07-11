import os
import pickle
from time import time
from datetime import datetime
from functools import partial

import numpy as np
import sonnet as snt
from tqdm import tqdm
import tensorflow as tf
from graph_nets.utils_tf import specs_from_graphs_tuple
from gn_contrib.train import binary_crossentropy

from rgt.utils import init_generator, get_accuracy

__all__ = ["EstimatorRGT"]


class EstimatorRGT(snt.Module):
    def __init__(
        self,
        rgt,
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
        input_fields=None,
        class_weights=[1.0, 1.0],
        scaler=True,
        delta_time_validation=60,
        log_path="logs",
        restore_path=None,
        compile=False,
        debug=False,
    ):
        np.random.seed(seed)
        tf.random.set_seed(seed)
        self._rs = np.random.RandomState(seed)
        super(EstimatorRGT, self).__init__(name="EstimatorRGT")

        self._best_acc = 0
        self._best_delta = np.infty
        self._delta_time_validation = delta_time_validation

        self._model = rgt
        self._tr_size = tr_size
        self._num_epochs = num_epochs
        self._tr_batch_size = tr_batch_size
        self._val_batch_size = val_batch_size
        self._loss_fn = partial(
            binary_crossentropy,
            entity="edges",
            class_weights=class_weights,
            msg_ratio=msg_ratio,
        )

        self._lr = tf.Variable(init_lr, trainable=False, dtype=tf.float32, name="lr")
        self._step = tf.Variable(0, trainable=False, dtype=tf.float32, name="tr_step")
        self._opt = snt.optimizers.__getattribute__(optimizer)(learning_rate=self._lr)
        self._schedule_lr_fn = tf.keras.optimizers.schedules.PolinomialDecay(
            init_lr, decay_steps, end_lr, power=power, cycle=cycle
        )

        self._tr_path_data = tr_path_data
        self._val_path_size = val_path_data
        self._input_fields = input_fields
        self.batch_generator = partial(
            init_generator,
            scaler=scaler,
            random_state=self._rs,
            file_ext=file_ext,
            input_fields=input_fields,
        )

        if restore_path is None:
            self._log_dir = os.path.join(
                log_path, datetime.now().strftime("%Y%m%d-%H%M%S")
            )
        else:
            self._log_dir = os.path.join(log_path, restore_path)
        self.__set_managers(seed, restore_path is None)

        if debug:
            tf.debugging.experimental.enable_dump_debug_info(
                dump_root=os.path.join(self._log_dir, "debug"),
                tensor_debug_mode="FULL_HEALTH",
                circular_buffer_size=-1,
            )

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

    def __save_random_state(self):
        with open(os.path.join(self._log_dir, "random_state.pkl"), "wb") as f:
            self._random_state = pickle.dump(self._random_state, f)

    def __set_managers(self, seed, restore):
        if restore:
            with open(os.path.join(self._log_dir, "seed.csv"), "r") as f:
                seed = int(f.readline().rstrip())
            with open(os.path.join(self._log_dir, "random_state.pkl"), "rb") as f:
                self._random_state = pickle.load(f)
            with open(os.path.join(self._log_dir, "stopped_step.csv"), "r") as f:
                try:
                    self.init__epoch, self._seen_graphs = list(
                        map(lambda s: int(s), f.readline().rstrip().split(","))
                    )
                except:
                    raise ValueError("Session restore is not possible")
        else:
            self._init_epoch = 0
            self._seen_graphs = 0
            with open(os.path.join(self._log_dir, "random_state.csv"), "w") as f:
                seed = f.write("{}\n".format(seed))
            self._random_state = np.random.RandomState(seed)

        tf.random.set_seed(seed)
        self._writer_scalars = tf.summary.create_file_writer(
            os.path.join(self._log_dir, "scalars")
        )
        ckpt = tf.train.Checkpoint(
            step=self._step,
            optimizer=self._opt,
            model=self._model,
            best_acc=self._best_acc,
            best_delta=self._best_delta,
        )
        self._last_ckpt_manager = tf.train.CheckpointManager(
            ckpt, os.path.join(self._log_dir, "last_ckpts"), max_to_keep=1
        )
        self._best_acc_ckpt_manager = tf.train.CheckpointManager(
            ckpt, os.path.join(self._log_dir, "best_acc_ckpts"), max_to_keep=5
        )
        self._best_delta_ckpt_manager = tf.train.CheckpointManager(
            ckpt, os.path.join(self._log_dir, "best_delta_ckpts"), max_to_keep=5
        )
        if restore:
            _ = ckpt.restore(self._last_ckpt_manager.latest_checkpoint)
            print(
                "\nRestore training from {}\n\tEpoch : {}, "
                "Seen  Graphs : {}, Best Acc : {}, Best Delta : {}\n".format(
                    self._log_dir, self._epoch, self._seen_graphs
                )
            )

    def __update_model_weights(self, in_graphs, gt_graphs):
        loss_for_all_mps = []
        expected = gt_graphs.edges
        targets = in_graphs.globals
        with tf.GradientTape() as tape:
            output_graphs = self._model(in_graphs, targets, True)
            for predicted_graphs in output_graphs:
                predicted = predicted_graphs.edges
                loss_for_all_mps.append(self._loss_fn(expected, predicted))
            loss = tf.math.reduce_sum(tf.stack(loss_for_all_mps))
            loss = loss / len(output_graphs)
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._opt.apply(gradients, self._model.trainable_variables)
        self._step.assign_add(1)
        self._lr.assign(self._schedule_lr_fn(self._step))
        return output_graphs[-1], loss.numpy()

    def __eval(self, in_graphs):
        targets = in_graphs.globals
        output_graphs = self._model(in_graphs, targets, False)
        return output_graphs[-1]

    def __assess_val(self):
        acc = []
        loss = []
        val_generator = self._batch_generator(self._val_path_data, self._val_batch_size)
        for in_graphs, gt_graphs, _ in val_generator:
            out_graphs = self._eval(in_graphs)
            acc.append(
                get_accuracy(out_graphs[-1].edges.numpy(), gt_graphs.edges.numpy())
            )
            loss.append(self._loss_fn(out_graphs, gt_graphs.edges).numpy())
        return sum(acc) / len(acc), sum(loss) / len(loss)

    def __log_scalars(self, params):
        with self._writer_scalars.as_default():
            for name, value in params.items():
                tf.summary.scalar(name, data=value, step=tf.cast(self._step, tf.int64))
            self._writer_scalers.flush()

    def train(self):
        start_time = time()
        last_validation = start_time
        for epoch in range(self._init_epoch, self._num_epochs):
            self.__save_random_state()
            epoch_bar = tqdm(
                total=self._tr_size,
                initial=self._seen_graphs,
                desc="Processed Graphs",
                leave=False,
            )
            epoch_bar.set_postfix(epoch="{} / {}".format(epoch, self._num_epochs))
            tr_generator = self._batch_generator(
                self._tr_path_data, self._batch_size, size=self._tr_size
            )
            for in_graphs, gt_graphs, _ in tr_generator:
                tr_out_graphs, tr_loss = self._update_model_weights(
                    in_graphs, gt_graphs
                )
                delta_time = time() - last_validation
                if delta_time >= self._delta_time_validation:
                    last_validation = time()
                    tr_acc = get_accuracy(
                        tr_out_graphs.edges.numpy(), gt_graphs.edges.numpy()
                    )
                    val_acc, val_loss = self.__assess_val()
                    delta = np.abs(tr_loss, val_loss)
                    self.__log_scalars(
                        {
                            "tr_loss": tf.constant(tr_loss),
                            "tr_acc": tf.constant(tr_acc),
                            "val_loss": tf.constant(val_loss),
                            "val_acc": tf.constant(val_acc),
                            "delta": tf.constant(delta),
                            "lr": self._lr,
                        }
                    )
                    self._last_ckpt_manager.save()
                    if self._best_acc <= val_acc:
                        self._best_acc_ckpt_manager.save()
                    if self._best_delta_acc >= delta:
                        self._best_delta_ckpt_manager.save()
                    epoch_bar.set_postfix(
                        epoch="{} / {}".format(epoch, self._num_epochs),
                        tr_loss=tr_loss,
                        val_loss=val_loss,
                        best_acc=self._best_acc,
                        best_delta=self._best_delta,
                    )
                epoch_bar.update(in_graphs.n_node.shape[0])
            epoch_bar.close()
