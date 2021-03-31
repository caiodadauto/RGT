from functools import partial

import sonnet as snt
import tensorflow as tf
from graph_nets import utils_tf
from graph_nets.modules import GraphIndependent

from routergn.nn import utils
from routergn.nn.modules import GraphRouter, GraphTopologyTranformer
from routergn.nn.models import (
    make_edge_encoder_routing,
    make_edge_routing,
    make_edge_tau,
    make_layer_norm,
    make_leaky_relu_mlp,
    make_node_tau,
)


__all__ = ["RoutingGraphTransformer"]


class RoutingGraphTransformer(snt.Module):
    def __init__(
        self,
        num_of_msg,
        num_of_heads_core,
        num_of_heads_routing,
        edge_independent_fn=make_leaky_relu_mlp,
        node_independent_fn=make_leaky_relu_mlp,
        edge_gtt_fn=make_edge_tau,
        node_gtt_fn=make_node_tau,
        edge_gr_fn=make_edge_routing,
        edge_encoder_routing_fn=make_edge_encoder_routing,
        edge_independent_kwargs=None,
        node_independent_kwargs=None,
        edge_gtt_kwargs=None,
        node_gtt_kwargs=None,
        edge_gr_kwargs=None,
        layer_norm_kwargs=None,
        name="RoutingGraphTransformer",
    ):
        super(RoutingGraphTransformer, self).__init__(name=name)
        default_independent_hs = 16
        default_independent_nl = 4
        default_norm_axis = -1
        default_gtt_hs = 16
        default_gtt_nl = 4
        default_gr_hs = 12
        default_gr_nl = 4
        default_independent_kwargs = dict(
            hidden_size=default_independent_hs, num_of_layers=default_independent_nl
        )
        default_norm_kwargs = dict(axis=default_norm_axis)
        default_edge_gtt_kwargs = dict(
            key_hidden_size=default_gtt_hs,
            key_num_of_layers=default_gtt_nl,
            query_hidden_size=default_gtt_hs,
            query_num_of_layers=default_gtt_nl,
            value_hidden_size=default_gtt_hs,
            value_num_of_layers=default_gtt_nl,
        )
        default_node_gtt_kwargs = dict(
            value_hidden_size=default_gtt_hs,
            value_num_of_layers=default_gtt_nl,
        )
        default_gr_kwargs = dict(
            key_hidden_size=default_gr_hs,
            key_num_of_layers=default_gr_nl,
            query_hidden_size=default_gr_hs,
            query_num_of_layers=default_gr_nl,
            value_hidden_size=default_gr_hs,
            value_num_of_layers=default_gr_nl,
        )
        if edge_independent_kwargs is None:
            edge_independent_kwargs = default_independent_kwargs
        if node_independent_kwargs is None:
            node_independent_kwargs = default_independent_kwargs
        if edge_gtt_kwargs is None:
            edge_gtt_kwargs = default_edge_gtt_kwargs
        if node_gtt_kwargs is None:
            node_gtt_kwargs = default_node_gtt_kwargs
        if edge_gr_kwargs is None:
            edge_gr_kwargs = default_gr_kwargs
        if layer_norm_kwargs is None:
            layer_norm_kwargs = default_norm_kwargs

        self._num_of_msg = num_of_msg
        edge_independent_fn = partial(edge_independent_fn, **edge_independent_kwargs)
        node_independent_fn = partial(node_independent_fn, **node_independent_kwargs)
        edge_gtt_fn = partial(edge_gtt_fn, **edge_gtt_kwargs)
        node_gtt_fn = partial(node_gtt_fn, **node_gtt_kwargs)
        edge_gr_fn = partial(edge_gr_fn, **edge_gr_kwargs)
        layer_norm_fn = partial(make_layer_norm, **layer_norm_kwargs)
        edge_encoder_routing_fn=partial(edge_encoder_routing_fn, **edge_independent_kwargs)

        self._encoder = GraphIndependent(
            edge_model_fn=edge_independent_fn, node_model_fn=node_independent_fn
        )

        self._gtt_heads = []
        for _ in range(num_of_heads_core):
            self._gtt_heads.append(
                GraphTopologyTranformer(
                    edge_model_fn=edge_gtt_fn, node_model_fn=node_gtt_fn
                )
            )
        self._encoder_multi_head_gtt = GraphIndependent(
            edge_model_fn=edge_independent_fn, node_model_fn=node_independent_fn
        )
        self._layer_norm_gtt = GraphIndependent(
            edge_model_fn=layer_norm_fn, node_model_fn=layer_norm_fn
        )

        self._gr_heads = []
        for _ in range(num_of_heads_routing):
            self._gr_heads.append(GraphRouter(edge_model_fn=edge_gr_fn))
        self._encoder_multi_head_gr = GraphIndependent(
            edge_model_fn=edge_encoder_routing_fn
        )

    def update_num_of_msg(self, num_of_msg):
        self._num_of_msg = num_of_msg

    def spread_msg(self, graphs, kwargs):
        all_graphs_out = []
        graphs_out = self._encoder(
            graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
        )
        for _ in range(self._num_of_msg):
            heads_out = []
            for gtt in self._gtt_heads:
                heads_out.append(
                    gtt(graphs_out, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
                )
            heads_out = utils_tf.concat(heads_out)
            heads_out = self._encoder_multi_head_gtt(heads_out)
            residual_connection = utils.sum([heads_out, graphs_out])
            graphs_out = self._layer_norm_gtt(residual_connection)
            all_graphs_out.append(graphs_out.copy())
        return all_graphs_out

    def route(self, graphs, kwargs):
        heads_out = []
        for gr in self._gr_heads:
            heads_out.append(gr(graphs, edge_model_kwargs=kwargs))
        heads_out = utils_tf.concat(heads_out)
        _ = kwargs.pop("is_training")
        _ = kwargs.pop("target")
        graphs_out = self._encoder_multi_head_gr(heads_out, edge_model_kwargs=kwargs)
        return graphs_out

    def __call__(self, graphs, targets, is_training):
        output = []
        repted_targets = utils_tf.repeat(targets, graphs.n_edge)
        kwargs = {"is_training": is_training}
        senders = graphs.senders
        num_of_nodes = tf.math.reduce_sum(graphs.n_node)
        all_graphs_out = self.spread_msg(graphs, kwargs)
        kwargs.update(target=repted_targets, senders=senders, num_of_nodes=num_of_nodes)
        for graphs_out in all_graphs_out:
            output.append(self.route(graphs_out, edge_model_kwargs=kwargs))
        return output
