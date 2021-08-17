import sonnet as snt
import tensorflow as tf
from functools import partial

from gn_contrib.snt_modules import (
    EdgeTau,
    make_leaky_relu_mlp,
)

all = [
    "EdgeRouting",
    "EdgeEncoderRouting",
    "make_conv_ip",
    "make_edge_encoder_routing",
    "make_edge_routing",
]


class EdgeEncoderRouting(snt.Module):
    def __init__(self, edge_model_fn, name="EdgeEncoderRouting"):
        super(EdgeEncoderRouting, self).__init__(name=name)
        self._edge_model = edge_model_fn()

    def _sent_edges_softmax(self, data, senders, num_nodes):
        denominator = tf.math.unsorted_segment_sum(data, senders, num_nodes)
        return data / tf.gather(denominator, senders)

    def __call__(self, inputs, is_training, senders, num_nodes):
        logist_out = tf.math.exp(self._edge_model(inputs, is_training))
        return self._sent_edges_softmax(logist_out, senders, num_nodes)


class EdgeRouting(EdgeTau):
    def _sent_edges_softmax(self, data, senders, num_nodes):
        denominator = tf.math.unsorted_segment_sum(data, senders, num_nodes)
        return data / tf.gather(denominator, senders)

    def __call__(self, inputs, target, senders, num_nodes, is_training):
        query = self._query_model(target, is_training)
        key = self._key_model(inputs, is_training)
        value = self._value_model(inputs, is_training)
        alpha = tf.math.exp(
            tf.math.sigmoid(
                tf.math.reduce_sum(query * key, keepdims=True, axis=-1) / self._ratio
            )
        )
        logist_out = tf.math.exp(
            tf.math.tanh(
                tf.math.reduce_sum(query * (alpha * value), keepdims=True, axis=-1)
            )
        )
        return self._sent_edges_softmax(logist_out, senders, num_nodes)


class ConvTarget(snt.Module):
    def __init__(
        self,
        conv_output_channels,
        conv_kernel,
        conv_stride,
        conv_padding,
        pool_ksize,
        pool_stride,
        pool_pedding,
        scale=True,
        offset=True,
        name="ConvIP",
    ):
        super(ConvTarget, self).__init__(name=name)
        self._convs = []
        self._pooling = partial(
            tf.nn.max_pool1d,
            ksize=pool_ksize,
            strides=pool_stride,
            padding=pool_pedding,
        )

        for output_channels, kernel, stride, padding in zip(
            conv_output_channels, conv_kernel, conv_stride, conv_padding
        ):
            self._convs.append(
                snt.Conv2D(
                    output_channels=output_channels,
                    kernel_shape=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
        self._layer_norm = snt.LayerNorm(-1, scale, offset)

    def __call__(self, inputs):
        # Considering the that all IP bytes are concatenated
        outputs = tf.reshape(inputs, shape=(-1, 4, 2, 4))
        outputs = tf.transpose(outputs, perm=[0, 2, 3, 1])
        for conv in self._convs:
            outputs = conv(outputs)
            outputs = self._pooling(outputs)
        outputs = self._layer_norm(snt.flatten(outputs))
        return outputs


class ConvEdgeIP(ConvTarget):
    def __init__(
        self,
        conv_output_channels,
        conv_kernel,
        conv_stride,
        conv_padding,
        pool_ksize,
        pool_stride,
        pool_pedding,
        hidden_sizes,
        dropout_rate,
        alpha,
        scale=True,
        offset=True,
        name="ConvIP",
    ):
        super(ConvEdgeIP, self).__init__(
            conv_output_channels,
            conv_kernel,
            conv_stride,
            conv_padding,
            pool_ksize,
            pool_stride,
            pool_pedding,
            scale=scale,
            offset=offset,
            name=name,
        )
        self._mlp = make_leaky_relu_mlp(
            hidden_sizes=hidden_sizes, dropout_rate=dropout_rate, alpha=alpha
        )
        self._layer_norm = snt.LayerNorm(-1, scale, offset)

    def __call__(self, inputs, **kwargs):
        # Considering the that all IP bytes are concatenated
        outputs = tf.reshape(inputs[:, :-1], shape=(-1, 4, 2, 4))
        outputs = tf.transpose(outputs, perm=[0, 2, 3, 1])
        edge_weights = tf.expand_dims(inputs[:, -1], axis=-1)
        for conv in self._convs:
            outputs = conv(outputs)
            outputs = self._pooling(outputs)
        outputs = tf.concat([snt.flatten(outputs), edge_weights], axis=-1)
        outputs = self._layer_norm(outputs)
        outputs = self._mlp(outputs, **kwargs)
        return outputs


def make_edge_routing(
    key_hidden_sizes,
    value_hidden_sizes,
    query_conv_output_channels,
    query_conv_kernel,
    query_conv_stride,
    query_conv_padding,
    query_pool_ksize,
    query_pool_stride,
    query_pool_pedding,
    query_hidden_sizes,
    key_dropout_rate=0.32,
    key_alpha=0.2,
    value_dropout_rate=0.32,
    value_alpha=0.2,
    query_dropout_rate=0.32,
    query_alpha=0.2,
    query_scale=True,
    query_offset=True,
):
    key_model_fn = partial(
        make_leaky_relu_mlp,
        key_hidden_sizes,
        key_dropout_rate,
        key_alpha,
    )
    query_model_fn = partial(
        ConvTarget,
        query_conv_output_channels,
        query_conv_kernel,
        query_conv_stride,
        query_conv_padding,
        query_pool_ksize,
        query_pool_stride,
        query_pool_pedding,
        query_hidden_sizes,
        query_dropout_rate,
        query_alpha,
        query_scale,
        query_offset,
    )
    value_model_fn = partial(
        make_leaky_relu_mlp,
        value_hidden_sizes,
        value_dropout_rate,
        value_alpha,
    )
    return EdgeRouting(
        key_model_fn, query_model_fn, value_model_fn, 0, key_hidden_sizes[-1]
    )


def make_edge_encoder_routing(hidden_sizes, dropout_rate=0.32, alpha=0.2):
    edge_model_fn = partial(
        make_leaky_relu_mlp,
        hidden_sizes,
        dropout_rate,
        alpha,
    )
    return EdgeEncoderRouting(edge_model_fn)


def make_conv_ip(
    conv_output_channels,
    conv_kernel,
    conv_stride,
    conv_padding,
    pool_ksize,
    pool_stride,
    pool_pedding,
    hidden_size,
    dropout_rate,
    alpha,
    scale=True,
    offset=True,
):
    return ConvEdgeIP(
        conv_output_channels,
        conv_kernel,
        conv_stride,
        conv_padding,
        pool_ksize,
        pool_stride,
        pool_pedding,
        hidden_size,
        dropout_rate,
        alpha,
        scale,
        offset,
    )
