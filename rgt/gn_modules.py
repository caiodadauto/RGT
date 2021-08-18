from functools import partial

import sonnet as snt
import tensorflow as tf
from graph_nets.utils_tf import repeat
from graph_nets.modules import GraphIndependent

from rgt.snt_modules import (
    make_conv_ip,
    make_edge_encoder_routing,
    make_edge_routing,
)
import gn_contrib.utils as utils
from gn_contrib.blocks import EdgeBlock
from gn_contrib.gn_modules import GraphTopologyTranformer
from gn_contrib.snt_modules import (
    make_leaky_relu_mlp,
    make_layer_norm,
    make_edge_tau,
    make_node_tau,
)


__all__ = [
    "RoutingGraphTransformer",
    "GraphRouter",
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


class CoreTransformer(snt.Module):
    def __init__(
        self,
        num_heads,
        edge_model_kwargs,
        node_model_kwargs,
        embed_multi_head_kwargs,
        layer_norm_kwargs,
        edge_model_fn=make_edge_tau,
        node_model_fn=make_node_tau,
        layer_norm_fn=make_layer_norm,
        name="CoreTransformer",
    ):
        super(CoreTransformer, self).__init__(name=name)
        edge_model_fn = partial(edge_model_fn, **edge_model_kwargs)
        node_model_fn = partial(node_model_fn, **node_model_kwargs)
        layer_norm_fn = partial(make_layer_norm, **layer_norm_kwargs)

        self._heads = []
        for _ in range(num_heads):
            self._heads.append(
                GraphTopologyTranformer(
                    edge_model_fn=edge_model_fn, node_model_fn=node_model_fn
                )
            )
        self._embedding_multi_head = GraphIndependent(
            edge_model_fn=lambda: snt.Linear(
                embed_multi_head_kwargs["edge_dim"],
                name="edge_embedding_multi_head_core",
            ),
            node_model_fn=lambda: snt.Linear(
                embed_multi_head_kwargs["node_dim"],
                name="node_embedding_multi_head_core",
            ),
        )
        self._layer_norm = GraphIndependent(
            edge_model_fn=layer_norm_fn, node_model_fn=layer_norm_fn
        )

    def __call__(self, graphs, kwargs):
        heads_out = []
        for gtt in self._heads:
            heads_out.append(
                gtt(graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
            )
        heads_out = utils.concat(heads_out, axis=-1, use_globals=False)
        heads_out = self._embedding_multi_head(heads_out)
        residual_connection = utils.sum([heads_out, graphs], use_globals=False)
        return self._layer_norm(residual_connection)


class LinkTransformer(snt.Module):
    def __init__(
        self,
        num_heads,
        edge_model_kwargs,
        embed_multi_head_kwargs,
        edge_model_fn=make_edge_routing,
        name="LinkTransformer",
    ):
        super(LinkTransformer, self).__init__(name=name)
        edge_model_fn = partial(edge_model_fn, **edge_model_kwargs)
        embedding_model_fn = partial(
            make_edge_encoder_routing, **embed_multi_head_kwargs
        )

        self._heads = []
        for _ in range(num_heads):
            self._heads.append(GraphRouter(edge_model_fn=edge_model_fn))
        self._embedding_multi_head = GraphIndependent(edge_model_fn=embedding_model_fn)

    def __call__(self, graphs, kwargs):
        heads_out = []
        for gr in self._heads:
            heads_out.append(gr(graphs, edge_model_kwargs=kwargs))
        heads_out = utils.concat(heads_out, axis=-1, use_globals=False)
        graphs = self._embedding_multi_head(
            heads_out,
            edge_model_kwargs=dict(
                is_training=kwargs["is_training"],
                senders=kwargs["senders"],
                num_nodes=kwargs["num_nodes"],
            ),
        )
        return graphs


class RoutingGraphTransformer(snt.Module):
    def __init__(
        self,
        num_msg,
        num_heads_core,
        num_heads_routing,
        dept_core,
        dept_routing,
        edge_encoder_kwargs,
        node_encoder_kwargs,
        edge_gtt_kwargs,
        node_gtt_kwargs,
        edge_gr_kwargs,
        embed_core_multi_head_kwargs,
        embed_routing_multi_head_kwargs,
        layer_norm_gtt_kwargs,
        edge_independent_fn=make_conv_ip,
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
        layer_norm_gtt_fn = partial(make_layer_norm, **layer_norm_gtt_kwargs)

        self._num_msg = num_msg
        self._encoder = GraphIndependent(
            edge_model_fn=edge_independent_fn,
            node_model_fn=node_independent_fn,
        )
        self._core_transformers = []
        self._routing_transformers = []
        for _ in range(dept_core):
            self._core_transformers.append(
                CoreTransformer(
                    num_heads_core,
                    edge_gtt_kwargs,
                    node_gtt_kwargs,
                    embed_core_multi_head_kwargs,
                    layer_norm_gtt_kwargs,
                    edge_gtt_fn,
                    node_gtt_fn,
                    layer_norm_gtt_fn,
                )
            )
        for _ in range(dept_routing):
            self._routing_transformers.append(
                LinkTransformer(
                    num_heads_routing,
                    edge_gr_kwargs,
                    embed_routing_multi_head_kwargs,
                    edge_gr_fn,
                )
            )

    def update_num_msg(self, num_msg):
        self._num_msg = num_msg

    def spread_msg(self, graphs, kwargs):
        all_graphs_out = []
        graphs_out = self._encoder(
            graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
        )
        for _ in range(self._num_msg):
            for core_model in self._core_transformers:
                graphs_out = core_model(graphs_out, kwargs)
            all_graphs_out.append(graphs_out)
        return all_graphs_out

    def route(self, graphs, kwargs):
        graphs_out = graphs
        for routing_model in self._routing_transformers:
            graphs_out = routing_model(graphs_out, kwargs)
        return graphs_out

    def __call__(self, graphs, targets, is_training):
        output = []
        repeated_targets = repeat(targets, graphs.n_edge)
        kwargs = {"is_training": is_training}
        senders = graphs.senders
        num_nodes = tf.math.reduce_sum(graphs.n_node)
        all_graphs_out = self.spread_msg(graphs, kwargs)
        kwargs.update(target=repeated_targets, senders=senders, num_nodes=num_nodes)
        for graphs_out in all_graphs_out:
            output.append(self.route(graphs_out, kwargs))
        return output
