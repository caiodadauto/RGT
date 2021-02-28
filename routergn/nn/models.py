from functools import partial

import sonnet as snt
import tensorflow as tf


__all__ = [
    "LeakyReluMLP",
    "EdgeTau",
    "NodeTau",
    "make_leaky_relu_mlp",
    "make_edge_tau",
    "make_node_tau",
    "make_edge_routing",
    "make_layer_norm",
    "make_edge_encoder_routing",
]


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

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear in self._linear_layers:
            if is_training:
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
    def _initialize_feature_dimension(self, inputs):
        dim_concat = inputs.shape[-1]
        if dim_concat % 3 != 0:
            raise ValueError(
                "It is expected the concatenation of three"
                " entity features for edge feature"
            )
        self._dim_feature = dim_concat // 3

    def __call__(self, inputs, is_training):
        self._initialize_feature_dimension(inputs)
        predecessor_features = inputs[:, 0 : self._dim_feature]
        query = self._query_model(predecessor_features, is_training)
        key = self._key_model(inputs, is_training)
        value = self._value_model(inputs, is_training)
        alpha = tf.math.exp(tf.math.reduce_sum(query * key, axis=-1))
        return tf.concat([tf.reshape(alpha, (-1, 1)), value], axis=-1)


class NodeTau(snt.Module):
    def __init__(self, value_model_fn, name="NodeTau"):
        super(NodeTau, self).__init__(name=name)
        self._value_model = value_model_fn()

    def __call__(self, inputs, is_training):
        return self._value_model(inputs, is_training)


class EdgeRouting(EdgeTau):
    def _sent_edges_softmax(self, data, senders, num_of_nodes):
        denominator = tf.math.unsorted_segment_sum(data, senders, num_of_nodes)
        return data / tf.gather(denominator, senders)

    # TODO: The target needs to be repeted in order to reach all nodes
    def __call__(self, inputs, target, senders, num_of_nodes, is_training):
        query = self._query_model(target, is_training)
        key = self._key_model(inputs, is_training)
        value = self._value_model(inputs, is_training)
        alpha = tf.math.exp(tf.math.reduce_sum(query * key, axis=-1))
        logist_out = tf.math.exp(tf.math.reduce_sum(query * (alpha * value), axis=-1))
        return self._sent_edges_softmax(logist_out, senders, num_of_nodes)


class EdgeEncoderRouting(snt.Module):
    def __init__(self, name="EdgeEncoderRouting"):
        super(EdgeEncoderRouting, self).__init__(name=name)
        self._linear = snt.Linear(self._hidden_size)

    def _sent_edges_softmax(self, data, senders, num_of_nodes):
        denominator = tf.math.unsorted_segment_sum(data, senders, num_of_nodes)
        return data / tf.gather(denominator, senders)

    def __call__(self, inputs, senders, num_of_nodes):
        logist_out = self._linear(inputs)
        return self._sent_edges_softmax(logist_out, senders, num_of_nodes)


def make_leaky_relu_mlp(hidden_size, num_of_layers, dropout_rate, alpha):
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
        make_leaky_relu_mlp,
        key_hidden_size,
        key_num_of_layers,
        key_dropout_rate,
        key_alpha,
    )
    query_model_fn = partial(
        make_leaky_relu_mlp,
        query_hidden_size,
        query_num_of_layers,
        query_dropout_rate,
        query_alpha,
    )
    value_model_fn = partial(
        make_leaky_relu_mlp,
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
        make_leaky_relu_mlp,
        value_hidden_size,
        value_num_of_layers,
        value_dropout_rate,
        value_alpha,
    )
    return NodeTau(value_model_fn)


def make_edge_routing(
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
        make_leaky_relu_mlp,
        key_hidden_size,
        key_num_of_layers,
        key_dropout_rate,
        key_alpha,
    )
    query_model_fn = partial(
        make_leaky_relu_mlp,
        query_hidden_size,
        query_num_of_layers,
        query_dropout_rate,
        query_alpha,
    )
    value_model_fn = partial(
        make_leaky_relu_mlp,
        value_hidden_size,
        value_num_of_layers,
        value_dropout_rate,
        value_alpha,
    )
    return EdgeRouting(key_model_fn, query_model_fn, value_model_fn)


def make_layer_norm(axis, scale=True, offset=True):
    return snt.LayerNorm(axis, scale, offset)


def make_edge_encoder_routing():
    return EdgeEncoderRouting()
