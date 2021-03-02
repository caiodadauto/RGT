import numpy as np


def mask_neighbors(idx):
    unique_idx = np.unique(idx)
    bool_one_hot = idx == unique_idx
    # arr_one_hot = bool_one_hot.astype(np.int8)
    # neg_arr_one_hot = (~bool_one_hot).astype(np.int8)
    return bool_one_hot.T


def flow(
    node,
    target,
    edge_weights,
    neighbor_weights,
    receivers,
    prob_links,
    mask,
    last_node=None,
):
    def get_next_node(data, node_receivers):
        indices = np.argsort(data, order=["weight", "prob"])
        next_node_idx = indices[0]
        next_node = node_receivers[next_node_idx]
        return next_node, next_node_idx

    node_prob_links = -1 * prob_links[mask[node]]
    node_edge_weights = edge_weights[mask[node]]
    node_receivers = receivers[mask[node]]
    data = np.array(
        list(zip(neighbor_weights[node], node_prob_links)),
        dtype=[("weight", np.int32), ("prob", np.float32)],
    )
    next_node, next_node_idx = get_next_node(data, node_receivers)

    if next_node == last_node:
        return 0, 0, True
    if next_node == target:
        return 0, 0, False
    else:
        while True:
            cost, hops, is_reverse = flow(
                next_node,
                target,
                edge_weights,
                receivers,
                prob_links,
                mask,
                neighbor_weights,
                node,
            )
            if is_reverse:
                neighbor_weights[next_node_idx] += 1
                data["weight"][next_node_idx] += 1
            else:
                return cost + node_edge_weights[next_node_idx], hops + 1, False
            next_node, next_node_idx = get_next_node(data, node_receivers)


def reverse_link(graph, target, edge_weights, sources=None):
    prob_links = graph.edges
    senders = graph.senders
    if sources is None:
        sources = np.unique(senders)

    mask = mask_neighbors(senders)
    neighbor_weights = {}
    for node in senders:
        neighbor_weights[node] = np.zeros(mask[node].sum())

    metrics = {"cost": {}, "hops": {}}
    for node in sources:
        cost, hops, _ = flow(
            node, target, edge_weights, neighbor_weights, prob_links, mask
        )
        metrics["cost"][node] = cost
        metrics["hops"][node] = hops
    return metrics
