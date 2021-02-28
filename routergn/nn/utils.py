import tree
import tensorflow as tf
from graph_nets import utils_np
from graph_nets.graphs import NODES, EDGES, GLOBALS


__all__ = ["norm_values", "sum"]

def norm_values(data, segment_ids, num_segments, name=None):
    values = data[:, 1:]
    if values.shape[1] == 1:
        values = tf.reshape(values, (-1, 1))
    alpha = tf.reshape(data[:, 0], (-1, 1))
    unnormalized_sum = tf.math.unsorted_segment_sum(
        alpha * values, segment_ids, num_segments, name
    )
    norm_ratio = tf.math.unsorted_segment_sum(alpha, segment_ids, num_segments, name)
    return tf.divide(unnormalized_sum, norm_ratio)


def _nested_sum(input_graphs, field_name, axis):
    features_list = [getattr(gr, field_name) for gr in input_graphs if getattr(gr, field_name) is not None]
    if not features_list:
        return None

    if len(features_list) < len(input_graphs):
        raise ValueError("All graphs or no graphs must contain {} features.".format(field_name))

    name = "sum_" + field_name
    return tree.map_structure(lambda *x: tf.math.add_n(x, axis, name), *features_list)


def sum(input_graphs, name="graph_sum"):
    if not input_graphs:
        raise ValueError("List argument `input_graphs` is empty")
    utils_np._check_valid_sets_of_keys([gr._asdict() for gr in input_graphs])
    if len(input_graphs) == 1:
        return input_graphs[0]

    with tf.name_scope(name):
        nodes = _nested_sum(input_graphs, NODES, -1)
        edges = _nested_sum(input_graphs, EDGES, -1)
        globals_ = _nested_sum(input_graphs, GLOBALS, -1)

    output = input_graphs[0].replace(nodes=nodes, edges=edges, globals=globals_)
    return output
