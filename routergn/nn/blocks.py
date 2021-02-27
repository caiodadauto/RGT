import tensorflow as tf
import graph_nets.blocks as blocks
from graph_nets.blocks import (
    broadcast_globals_to_edges,
    broadcast_globals_to_nodes,
    broadcast_receiver_nodes_to_edges,
    broadcast_sender_nodes_to_edges,
    _get_static_num_edges,
    _get_static_num_nodes,
    _validate_graph,
    EDGES,
    NODES,
)


__all__ = ["EdgeBlock", "NodeBlock"]

# TODO: Assess if this concatenation modification has any performance improvement.
class EdgeBlock(blocks.EdgeBlock):
    def _collect_features(self, graph):
        """Collects the features of interest for edges"""
        edges_to_collect = []

        if self._use_receiver_nodes:
            edges_to_collect.append(broadcast_receiver_nodes_to_edges(graph))

        if self._use_edges:
            _validate_graph(graph, (EDGES,), "when use_edges == True")
            edges_to_collect.append(graph.edges)

        if self._use_sender_nodes:
            edges_to_collect.append(broadcast_sender_nodes_to_edges(graph))

        if self._use_globals:
            num_edges_hint = _get_static_num_edges(graph)
            edges_to_collect.append(
                broadcast_globals_to_edges(graph, num_edges_hint=num_edges_hint)
            )

        collected_edges = tf.concat(edges_to_collect, axis=-1)
        return collected_edges


class NodeBlock(blocks.NodeBlock):
    def _collect_features(self, graph):
        """Collects the features of interest for nodes"""
        nodes_to_collect = []

        if self._use_nodes:
            _validate_graph(graph, (NODES,), "when use_nodes == True")
            nodes_to_collect.append(graph.nodes)

        if self._use_received_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph))

        if self._use_sent_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph))

        if self._use_globals:
            # The hint will be an integer if the graph has node features and the total
            # number of nodes is known at tensorflow graph definition time, or None
            # otherwise.
            num_nodes_hint = _get_static_num_nodes(graph)
            nodes_to_collect.append(
                broadcast_globals_to_nodes(graph, num_nodes_hint=num_nodes_hint)
            )

        collected_nodes = tf.concat(nodes_to_collect, axis=-1)
        return collected_nodes
