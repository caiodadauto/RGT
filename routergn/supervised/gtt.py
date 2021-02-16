import numpy as np
import sonnet as snt
import tensorflow as tf
from graph_nets import GraphNetwork


def _make_leaky_relu_mlp(hidden_size, num_of_layers, dropout_rate=0.4, alpha=0.75):
    return LeakyReluMLP(hidden_size, num_of_layers, dropout_rate, alpha)


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


class EdgeTranformer(snt.Module):
    def __init__(self, key_model_fn, query_model_fn, value_model_fn, name="EdgeTransformer"):
        self._key_model = key_model_fn()
        self._query_model = query_model_fn()
        self._value_model = value_model_fn()

    def __call__(self, inputs):
        alpha = tf.math.dot(self._query_model(inputs), self._key)


class NodeTranformer(snt.Module):
    def __init__(self, key_model_fn, query_model_fn, value_model_fn, name="NodeTransformer"):
        self._key_model = key_model_fn()
        self._query_model = query_model_fn()
        self._value_model = value_model_fn()

    def __call__(self, inputs):
        pass


class GTT(snt.Module):
    def __init__(
        self,
        key_model_fn,
        query_model_fn,
        value_model_fn,
        name="GraphTopologyTranformer",
    ):
        super(GTT, self).__init__(name=name)

    def __call__(self, inputs):
        pass
