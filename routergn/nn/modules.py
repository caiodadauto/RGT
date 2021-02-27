import sonnet as snt

from routergn.nn.blocks import EdgeBlock, NodeBlock
from routergn.nn.utils import norm_values


__all__ = ["GraphTopologyTranformer"]


class GraphTopologyTranformer(snt.Module):
    def __init__(
        self,
        edge_model_fn,
        node_model_fn,
        reducer=norm_values,
        name="GraphTopologyTranformer",
    ):
        super(GraphTopologyTranformer, self).__init__(name=name)
        self._edge_block = EdgeBlock(edge_model_fn=edge_model_fn, use_globals=False)
        self._node_block = NodeBlock(
            node_model_fn=node_model_fn,
            use_globals=False,
            received_edges_reducer=reducer,
        )

    def __call__(self, graphs, edge_model_kwargs=None, node_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}
        if node_model_kwargs is None:
            node_model_kwargs = {}
        return self.node_blocks(
            self._edge_block(graphs, edge_model_kwargs=edge_model_kwargs),
            node_model_kwargs=node_model_kwargs,
        )


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
