from functools import partial

import sonnet as snt
import tensorfloe as tf
from graph_nets.modules import GraphIndependent

from routergn import nn


class RoutingGraphTransformer(snt.Models):
    def __init__(
        self,
        edge_encoder_kwargs=None,
        node_encoder_kwargs=None,
        edge_gtt_kwargs=None,
        node_gtt_kwargs=None,
        edge_gr_kwargs=None,
        name="RoutingGraphTransformer",
    ):
        super(RoutingGraphTransformer).__init__(name=name)
        for kwargs in [
            edge_encoder_kwargs,
            node_encoder_kwargs,
            edge_gtt_kwargs,
            node_gtt_kwargs,
            edge_gr_kwargs,
        ]:
            if kwargs is None:
                kwargs = {}

        edge_encoder_fn = partial(nn.models.make_leaky_relu_mlp, **edge_encoder_kwargs)
        node_encoder_fn = partial(nn.models.make_leaky_relu_mlp, **node_encoder_kwargs)
        edge_gtt_fn = partial(nn.models.make_edge_tau, **edge_gtt_kwargs)
        node_gtt_fn = partial(nn.models.make_edge_tau, **node_gtt_kwargs)
        edge_gr_fn = partial(nn.models.make_edge_routing, **edge_gr_kwargs)
        self._encoder = GraphIndependent(
            edge_model_fn=edge_encoder_fn, node_model_fn=node_encoder_fn
        )
        self._gtt = nn.modules.GraphTopologyTranformer(
            edge_model_fn=edge_gtt_fn, node_model_fn=node_gtt_fn
        )
        self._gr = nn.modules.GraphRouter(edge_model_fn=edge_gr_fn)

    def __call__(self, graphs, num_of_msg, target, is_training):
        kwargs = {"is_training": is_training}
        senders = graphs.senders
        num_of_nodes = tf.math.reduce_sum(graphs.n_node)
        graphs_out = self._encoder(graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
        for _ in range(num_of_msg):
            graphs_out = self._gtt(graphs_out, edge_model_kwargs=kwargs, node_model_kwargs=kwargs)
        kwargs.update(target=target, senders=senders, num_of_nodes=num_of_nodes)
        return self._gr(graphs_out, edge_model_kwargs=kwargs)
