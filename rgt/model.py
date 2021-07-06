from functools import partial

import sonnet as snt
import tensorflow as tf
from graph_nets.utils_tf import repeat
from graph_nets.modules import GraphIndependent

import gn_contrib.utils as utils
from gn_contrib.blocks import EdgeBlock
from gn_contrib.gn_modules import GraphTopologyTranformer
from gn_contrib.snt_modules import (
    EdgeTau,
    make_leaky_relu_mlp,
    make_layer_norm,
    make_edge_tau,
    make_node_tau,
)


__all__ = [
    "RoutingGraphTransformer",
    "GraphRouter",
    "EdgeRouting",
    "EdgeEncoderRouting",
]


class GraphRouter(snt.Module):
    def __init__(
        self,
        edge_model_fn,
        name="GraphRouter",
    ):
        super(GraphRouter, self).__init__(name=name)
        self._edge_block = EdgeBlock(edge_model_fn=edge_model_fn, use_globals=False)

    def __call__(self, graphs, edge_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}
        return self._edge_block(graphs, edge_model_kwargs=edge_model_kwargs)


class EdgeRouting(EdgeTau):
    def _sent_edges_softmax(self, data, senders, num_of_nodes):
        denominator = tf.math.unsorted_segment_sum(data, senders, num_of_nodes)
        return data / tf.gather(denominator, senders)

    def __call__(self, inputs, target, senders, num_of_nodes, is_training):
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
        # print("OUTPUT MULTIHEAD ROUTING ENCODER ===============================")
        # print(logist_out.numpy().sum())
        # print("END ===============================")
        return self._sent_edges_softmax(logist_out, senders, num_of_nodes)


class EdgeEncoderRouting(snt.Module):
    def __init__(self, edge_model_fn, name="EdgeEncoderRouting"):
        super(EdgeEncoderRouting, self).__init__(name=name)
        self._edge_model = edge_model_fn()

    def _sent_edges_softmax(self, data, senders, num_of_nodes):
        denominator = tf.math.unsorted_segment_sum(data, senders, num_of_nodes)
        return data / tf.gather(denominator, senders)

    def __call__(self, inputs, senders, num_of_nodes):
        logist_out = tf.math.exp(self._edge_model(inputs))
        return self._sent_edges_softmax(logist_out, senders, num_of_nodes)


def make_edge_routing(
    key_hidden_sizes,
    query_hidden_sizes,
    value_hidden_sizes,
    key_dropout_rate=0.32,
    key_alpha=0.2,
    query_dropout_rate=0.32,
    query_alpha=0.2,
    value_dropout_rate=0.32,
    value_alpha=0.2,
):
    key_model_fn = partial(
        make_leaky_relu_mlp,
        key_hidden_sizes,
        key_dropout_rate,
        key_alpha,
    )
    query_model_fn = partial(
        make_leaky_relu_mlp,
        query_hidden_sizes,
        query_dropout_rate,
        query_alpha,
    )
    value_model_fn = partial(
        make_leaky_relu_mlp,
        value_hidden_sizes,
        value_dropout_rate,
        value_alpha,
    )
    return EdgeRouting(key_model_fn, query_model_fn, value_model_fn)


def make_edge_encoder_routing(hidden_sizes, dropout_rate=0.32, alpha=0.2):
    edge_model_fn = partial(
        make_leaky_relu_mlp,
        hidden_sizes,
        dropout_rate,
        alpha,
    )
    return EdgeEncoderRouting(edge_model_fn)


class RoutingGraphTransformer(snt.Module):
    def __init__(
        self,
        num_of_msg,
        num_of_heads_core,
        num_of_heads_routing,
        edge_gr_kwargs,
        edge_gtt_kwargs,
        node_gtt_kwargs,
        edge_multi_head_kwargs,
        node_multi_head_kwargs,
        edge_encoder_kwargs,
        node_encoder_kwargs,
        layer_norm_kwargs,
        edge_independent_fn=make_leaky_relu_mlp,
        node_independent_fn=make_leaky_relu_mlp,
        edge_gtt_fn=make_edge_tau,
        node_gtt_fn=make_node_tau,
        edge_gr_fn=make_edge_routing,
        name="RoutingGraphTransformer",
    ):
        super(RoutingGraphTransformer, self).__init__(name=name)

        edge_independent_fn = partial(edge_independent_fn, **edge_encoder_kwargs)
        node_independent_fn = partial(node_independent_fn, **node_encoder_kwargs)
        edge_gtt_fn = partial(edge_gtt_fn, **edge_gtt_kwargs)
        node_gtt_fn = partial(node_gtt_fn, **node_gtt_kwargs)
        edge_gr_fn = partial(edge_gr_fn, **edge_gr_kwargs)
        layer_norm_fn = partial(make_layer_norm, **layer_norm_kwargs)

        self._gr_heads = []
        self._gtt_heads = []
        self._num_of_msg = num_of_msg
        self._encoder = GraphIndependent(
            edge_model_fn=edge_independent_fn,
            node_model_fn=node_independent_fn,
        )
        for _ in range(num_of_heads_core):
            self._gtt_heads.append(
                GraphTopologyTranformer(
                    edge_model_fn=edge_gtt_fn, node_model_fn=node_gtt_fn
                )
            )
        for _ in range(num_of_heads_routing):
            self._gr_heads.append(GraphRouter(edge_model_fn=edge_gr_fn))
        self._encoder_multi_head_gtt = GraphIndependent(
            edge_model_fn=lambda: snt.Linear(
                edge_multi_head_kwargs["gtt_hidden_size"], name="edge_encoder_gtt"
            ),
            node_model_fn=lambda: snt.Linear(
                node_multi_head_kwargs["gtt_hidden_size"], name="node_encoder_gtt"
            ),
        )
        self._encoder_multi_head_gr = GraphIndependent(edge_model_fn=EdgeEncoderRouting)
        self._layer_norm_gtt = GraphIndependent(
            edge_model_fn=layer_norm_fn, node_model_fn=layer_norm_fn
        )

    def update_num_of_msg(self, num_of_msg):
        self._num_of_msg = num_of_msg

    def spread_msg(self, graphs, kwargs):
        all_graphs_out = []
        graphs_out = self._encoder(
            graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
        )
        # print("After Encoder Input ===============================")
        # print(graphs_out.edges.shape)
        # print(graphs_out.edges.numpy().sum())
        # print("END ===============================")
        for _ in range(self._num_of_msg):
            heads_out = []
            for gtt in self._gtt_heads:
                heads_out.append(
                    gtt(graphs_out, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
                )
            heads_out = utils.concat(heads_out, axis=-1, use_globals=False)
            # print("After Concat <MSG {}>===============================".format(i))
            # print(heads_out.edges.shape)
            # print(heads_out.edges.numpy().sum())
            # print("END ===============================")
            heads_out = self._encoder_multi_head_gtt(heads_out)
            # print("After Encoder <MSG {}>===============================".format(i))
            # print(heads_out.edges.shape)
            # print(heads_out.edges.numpy().sum())
            # print("END ===============================")
            residual_connection = utils.sum([heads_out, graphs_out], use_globals=False)
            # print("After Residual <MSG {}>===============================".format(i))
            # print(residual_connection.edges.shape)
            # print(residual_connection.edges.numpy().sum())
            # print("END ===============================")
            all_graphs_out.append(self._layer_norm_gtt(residual_connection))
            # print(i)
            graphs_out = all_graphs_out[-1]
        return all_graphs_out

    def route(self, graphs, kwargs):
        heads_out = []
        for gr in self._gr_heads:
            heads_out.append(gr(graphs, edge_model_kwargs=kwargs))
        heads_out = utils.concat(heads_out, axis=-1, use_globals=False)
        # print("After Concat <ROUTING>===============================")
        # print(heads_out.edges.shape)
        # print(heads_out.edges.numpy().sum())
        # print("END ===============================")
        graphs_out = self._encoder_multi_head_gr(
            heads_out,
            edge_model_kwargs=dict(
                senders=kwargs["senders"],
                num_of_nodes=kwargs["num_of_nodes"],
            ),
        )
        # print("After Encoder <ROUTING>===============================")
        # print(graphs_out.edges.shape)
        # print(graphs_out.edges.numpy().sum())
        # print("END ===============================")
        return graphs_out

    def __call__(self, graphs, targets, is_training):
        output = []
        repeated_targets = repeat(targets, graphs.n_edge)
        kwargs = {"is_training": is_training}
        senders = graphs.senders
        num_of_nodes = tf.math.reduce_sum(graphs.n_node)
        all_graphs_out = self.spread_msg(graphs, kwargs)
        kwargs.update(
            target=repeated_targets, senders=senders, num_of_nodes=num_of_nodes
        )
        for graphs_out in all_graphs_out:
            output.append(self.route(graphs_out, kwargs))
        return output
