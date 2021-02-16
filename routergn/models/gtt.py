from functools import partial

import numpy as np
import sonnet as snt
import tensorflow as tf
from graph_nets import GraphNetwork


def _make_leaky_relu_mlp(hidden_size, num_of_layers, dropout_rate, alpha):
    return LeakyReluMLP(hidden_size, num_of_layers, dropout_rate, alpha)


def make_edge_tau(
    key_hidden_size,
    key_num_of_layers,
    query_hidden_size,
    query_num_of_layers,
    value_hidden_size,
    value_num_of_layers,
    key_dropout_rate=0.4,
    key_alpha=0.2,
    query_dropout_rate=0.4,
    query_alpha=0.2,
    value_dropout_rate=0.4,
    value_alpha=0.2,
):
    key_model_fn = partial(
        _make_leaky_relu_mlp,
        key_hidden_size,
        key_num_of_layers,
        key_dropout_rate,
        key_alpha,
    )
    query_model_fn = partial(
        _make_leaky_relu_mlp,
        query_hidden_size,
        query_num_of_layers,
        query_dropout_rate,
        query_alpha,
    )
    value_model_fn = partial(
        _make_leaky_relu_mlp,
        value_hidden_size,
        value_num_of_layers,
        value_dropout_rate,
        value_alpha,
    )
    return EdgeTau(key_model_fn, query_model_fn, value_model_fn)


def make_node_tau(
    value_hidden_size, value_num_of_layers, value_dropout_rate=0.4, value_alpha=0.2
):
    value_model_fn = partial(
        _make_leaky_relu_mlp,
        value_hidden_size,
        value_num_of_layers,
        value_dropout_rate,
        value_alpha,
    )
    return NodeTau(value_model_fn)


class LeakyReluMLP(snt.Module):
    def __init__(
        self, hidden_size, num_of_layers, dropout_rate, alpha, name="LeakyReluMLP"
    ):
        super(LeakyReluMLP, self).__init__(name=name)
        self._linear_layers = []
        self._alpha = alpha
        self._hidden_size = hidden_size
        self._dropout_rate = dropout_rate
        self._num_of_layers = num_of_layers
        for _ in range(self._num_of_extra_layers):
            self._linear_layers.append(snt.Linear(self._hidden_size))

    def __call__(self, inputs):
        outputs_op = inputs
        for linear in self._linear_layers:
            outputs_op = tf.nn.dropout(inputs, rate=self._dropout_rate)
            outputs_op = linear(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=self._alpha)
        return outputs_op


class EdgeTau(snt.Module):
    def __init__(self, key_model_fn, query_model_fn, value_model_fn, name="EdgeTau"):
        super(EdgeTau, self).__init__(name=name)
        self._key_model = key_model_fn()
        self._query_model = query_model_fn()
        self._value_model = value_model_fn()

    @snt.once
    def _initialize_feature_dimension(inputs):
        dim_concat = inputs.shape[-1]
        if dim_concat % 3 != 0:
            raise ValueError(
                "It is expected the concatenation of three"
                " entity features for edge feature"
            )
        self._dim_feature = dim_concat // 3

    def __call__(self, inputs):
        self._initialize_feature_dimension(inputs)
        predecessor_features = inputs[:, 0 : self._dim_feature]
        query = self._query_model(predecessor_features)
        key = self._key_model(inputs)
        value = self._value_model(inputs)
        alpha = tf.math.reduce_sum(query * key, axis=-1)
        return tf.concat([tf.reshape(alpha, (-1, 1)), value], axis=-1)


class NodeTau(snt.Module):
    def __init__(self, value_model_fn, name="NodeTau"):
        super(NodeTau, self).__init__(name=name)
        self._value_model = value_model_fn()

    def __call__(self, inputs):
        return self._value_model(inputs)


def norm_values(data, segment_ids, num_segments, name=None):
    values = data[:, 1:]
    if values.shape[1] == 1:
        values = tf.reshape(values, (-1, 1))
    alpha = tf.reshape(data[:, 0], (-1, 1))
    unnormalized_sum = tf.math.unsorted_segment_sum(
        alpha * values, segment_ids, num_segments, name
    )
    norm_ratio = tf.math.unsorted_segment_sum(alpha, segment_ids, num_segments, name)
    return tf.divide(unnormalized_sum, norm_ratio)


class GraphTopologyTranformer(snt.Module):
    def __init__(
        self,
        node_model_fn,
        edge_model_fn,
        reducer=norm_values,
        name="GraphTopologyTranformer",
    ):
        super(GraphTopologyTranformer, self).__init__(name=name)

    def __call__(self, inputs):
        pass
