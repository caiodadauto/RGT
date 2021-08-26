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
        print(query.shape)
        print(key.shape)
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
        conv_dropout_rate,
        pool_ksize,
        pool_stride,
        pool_padding,
        scale=True,
        offset=True,
        name="ConvIP",
    ):
        super(ConvTarget, self).__init__(name=name)
        self._convs = []
        self._max_pools = []
        for output_channels, kernel, stride, padding in zip(
            conv_output_channels, conv_kernel, conv_stride, conv_padding
        ):
            self._convs.append(
                snt.Conv1D(
                    output_channels=output_channels,
                    kernel_shape=kernel,
                    stride=stride,
                    padding=padding,
                )
            )
        if pool_ksize is None or pool_stride is None or pool_padding is None:
            for _ in range(len(self._convs)):
                self._max_pools.append(lambda x: x)
        else:
            for ksize, stride, padding in zip(pool_ksize, pool_stride, pool_padding):
                if ksize is None or stride is None or padding is None:
                    self._max_pools.append(lambda x: x)
                else:
                    self._max_pools.append(
                        partial(
                            tf.nn.max_pool1d,
                            ksize=ksize,
                            strides=stride,
                            padding=padding,
                        )
                    )
        self._dropout_rate = conv_dropout_rate
        self._layer_norm = snt.LayerNorm(-1, scale, offset)

    def __call__(self, inputs, is_training):
        # Considering the that all IP bytes are concatenated
        outputs = tf.expand_dims(inputs, axis=-1)
        for conv, max_pool in zip(self._convs, self._max_pools):
            outputs = conv(outputs)
            outputs = max_pool(outputs)
            if is_training:
                outputs = tf.nn.dropout(outputs, rate=self._dropout_rate)
        outputs = self._layer_norm(snt.flatten(outputs))
        print(outputs.shape)
        # FIXME: (340, 256) wrong shape
        return outputs


class ConvEdgeIP(ConvTarget):
    def __init__(
        self,
        conv_output_channels,
        conv_kernel,
        conv_stride,
        conv_padding,
        conv_dropout_rate,
        pool_ksize,
        pool_stride,
        pool_padding,
        mlp_hidden_sizes,
        mlp_dropout_rate,
        mlp_alpha,
        scale=True,
        offset=True,
        name="ConvIP",
    ):
        super(ConvEdgeIP, self).__init__(
            conv_output_channels,
            conv_kernel,
            conv_stride,
            conv_padding,
            conv_dropout_rate,
            pool_ksize,
            pool_stride,
            pool_padding,
            scale=scale,
            offset=offset,
            name=name,
        )
        self._mlp = make_leaky_relu_mlp(
            hidden_sizes=mlp_hidden_sizes,
            dropout_rate=mlp_dropout_rate,
            alpha=mlp_alpha,
        )
        self._layer_norm = snt.LayerNorm(-1, scale, offset)

    def __call__(self, inputs, is_training):
        # Considering the that all IP bytes are concatenated
        outputs = tf.expand_dims(inputs[:, :-1], axis=-1)
        edge_weights = tf.expand_dims(inputs[:, -1], axis=-1)
        for conv, max_pool in zip(self._convs, self._max_pools):
            outputs = conv(outputs)
            outputs = max_pool(outputs)
            if is_training:
                outputs = tf.nn.dropout(outputs, rate=self._dropout_rate)
        outputs = tf.concat([snt.flatten(outputs), edge_weights], axis=-1)
        outputs = self._layer_norm(snt.flatten(outputs))
        outputs = self._mlp(outputs, is_training)
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
    query_pool_padding,
    key_dropout_rate=0.32,
    key_alpha=0.2,
    value_dropout_rate=0.32,
    value_alpha=0.2,
    query_scale=True,
    query_offset=True,
    query_conv_dropout_rate=0.25,
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
        query_conv_dropout_rate,
        query_pool_ksize,
        query_pool_stride,
        query_pool_padding,
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
    conv_dropout_rate,
    pool_ksize,
    pool_stride,
    pool_padding,
    mlp_hidden_sizes,
    mlp_dropout_rate,
    mlp_alpha,
    scale=True,
    offset=True,
):
    return ConvEdgeIP(
        conv_output_channels,
        conv_kernel,
        conv_stride,
        conv_padding,
        conv_dropout_rate,
        pool_ksize,
        pool_stride,
        pool_padding,
        mlp_hidden_sizes,
        mlp_dropout_rate,
        mlp_alpha,
        scale,
        offset,
    )
