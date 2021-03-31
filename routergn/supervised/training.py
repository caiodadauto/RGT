from time import time
from functools import partial

import tensorflow as tf
from tensorflow.math import log, reduce_mean


__all__ = ["train"]


def binary_cross_entropy(expected, predicted, class_weights):
    epsilon = 1e-7
    losses = -1 * (
        expected * log(predicted + epsilon)
        + (1 - expected) * log(1 - predicted + epsilon)
    )
    losses = tf.gather(class_weights, expected) * losses
    return reduce_mean(losses)


def update_weights(model, opt, in_graphs, gt_graphs, class_weights):
    losses = []
    expected = gt_graphs.edges
    with tf.GradientTape() as tape:
        output_graphs = model(in_graphs)
        for predicted_graphs in output_graphs:
            losses.append(
                binary_cross_entropy(expected, predicted_graphs.edges, class_weights)
            )
        loss_value = tf.math.reduce_sum(tf.stack(losses)) / len(output_graphs)
    gradients = tape.gradient(loss_value, model.trainable_variables)
    opt.apply(gradients, model.trainable_variables)
    return output_graphs, loss_value


def log_params(output_graphs, ground_truth_graphs, loss_value, epoch, n_batch):
    print(loss_value, epoch, n_batch)


def train(
    model,
    opt,
    graph_batch_generator,
    # validation_graphs,
    num_of_epochs,
    class_weights=tf.constant([1.0, 1.0]),
    delta_log=30,
):
    last_log_time = time()
    for epoch in range(num_of_epochs):
        n_batch = 0
        for input_graphs, ground_truth_graphs in graph_batch_generator:
            output_graphs, loss_value = update_weights(
                model, opt, input_graphs, ground_truth_graphs, class_weights
            )
            delta_time = time() - last_log_time
            if delta_time >= delta_log:
                log_params(
                    output_graphs, ground_truth_graphs, loss_value, epoch, n_batch
                )
                last_log_time = time()
            n_batch += 1
        log_params(output_graphs, ground_truth_graphs, loss_value, epoch, n_batch)
