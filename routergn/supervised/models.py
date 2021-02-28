from functools import partial

import sonnet as snt
import tensorfloe as tf
from graph_nets import utils_tf
from graph_nets.modules import GraphIndependent

from routergn import nn
from routergn import utils


class RoutingGraphTransformer(snt.Models):
    def __init__(
        self,
        num_of_msg,
        num_of_heads_core,
        num_of_heads_routing,
        edge_independent_fn,
        node_independent_fn,
        edge_gtt_fn,
        node_gtt_fn,
        edge_gr_fn,
        edge_independent_kwargs=None,
        node_independent_kwargs=None,
        edge_gtt_kwargs=None,
        node_gtt_kwargs=None,
        edge_gr_kwargs=None,
        layer_norm_kwargs=None,
        name="RoutingGraphTransformer",
    ):
        super(RoutingGraphTransformer).__init__(name=name)
        for kwargs in [
            edge_independent_kwargs,
            node_independent_kwargs,
            edge_gtt_kwargs,
            node_gtt_kwargs,
            edge_gr_kwargs,
            layer_norm_kwargs,
        ]:
            if kwargs is None:
                kwargs = {}

        self._num_of_msg = num_of_msg
        edge_independent_fn = partial(edge_independent_fn, **edge_independent_kwargs)
        node_independent_fn = partial(node_independent_fn, **node_independent_kwargs)
        edge_gtt_fn = partial(edge_gtt_fn, **edge_gtt_kwargs)
        node_gtt_fn = partial(node_gtt_fn, **node_gtt_kwargs)
        edge_gr_fn = partial(edge_gr_fn, **edge_gr_kwargs)
        layer_norm_fn = partial(nn.models.make_layer_norm, **layer_norm_kwargs)

        self._encoder = GraphIndependent(
            edge_model_fn=edge_independent_fn, node_model_fn=node_independent_fn
        )

        self._gtt_heads = []
        for _ in range(num_of_heads_core):
            self._gtt_heads += nn.modules.GraphTopologyTranformer(
                edge_model_fn=edge_gtt_fn, node_model_fn=node_gtt_fn
            )
        self._encoder_multi_head_gtt = GraphIndependent(
            edge_model_fn=edge_independent_fn, node_model_fn=node_independent_fn
        )
        self._layer_norm_gtt = GraphIndependent(
            edge_model_fn=layer_norm_fn, node_model_fn=layer_norm_fn
        )

        self._gr_heads = []
        for _ in range(num_of_heads_routing):
            self._gr_heads += nn.modules.GraphRouter(edge_model_fn=edge_gr_fn)
        self._encoder_multi_head_gr = GraphIndependent(edge_model_fn=nn.models.make_edge_encoder_routing)

    def update_num_of_msg(self, num_of_msg):
        self._num_of_msg = num_of_msg

    def spread_msg(self, graphs, kwargs):
        graphs_out = self._encoder(graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
        for _ in range(self._num_of_msg):
            heads_out = []
            for gtt in self._gtt_heads:
                heads_out += gtt(graphs_out, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
            heads_out = utils_tf.concat(heads_out)
            heads_out = self._encoder_multi_head_gtt(heads_out)
            graphs_out = self._layer_norm_gtt(utils.sum((heads_out, graphs_out)))
        return graphs_out

    def route(self, graphs, kwargs):
        heads_out = []
        for gr in self._gr_heads:
            heads_out += gr(graphs, edge_model_kwargs=kwargs)
        heads_out = utils_tf.concat(heads_out)
        _ = kwargs.pop("is_training")
        _ = kwargs.pop("target")
        graphs_out = self._encoder_multi_head_gr(heads_out, edge_model_kwargs=kwargs)
        return graphs_out

    def __call__(self, graphs, target, is_training):
        kwargs = {"is_training": is_training}
        senders = graphs.senders
        num_of_nodes = tf.math.reduce_sum(graphs.n_node)
        graphs_out = self.spread_msg(graphs, kwargs)
        kwargs.update(target=target, senders=senders, num_of_nodes=num_of_nodes)
        return self.route(graphs_out, edge_model_kwargs=kwargs)

