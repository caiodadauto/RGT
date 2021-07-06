import numpy as np

__all__ = ["reverse_link"]


def mask_neighbors(idx):
    unique_idx = np.unique(idx)
    mask = np.zeros((unique_idx.shape[0], idx.shape[0]), dtype=bool)
    for i in unique_idx:
        mask[i] = idx == i
    return mask


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

    local_cost = 0
    local_hops = 0
    node_prob_links = -1 * prob_links[mask[node]]
    node_edge_weights = edge_weights[mask[node]]
    node_receivers = receivers[mask[node]]
    data = np.array(
        list(zip(neighbor_weights[node], node_prob_links)),
        dtype=[("weight", np.int32), ("prob", np.float32)],
    )
    while True:
        next_node, next_node_idx = get_next_node(data, node_receivers)
        print("source:", node)
        print("neighbor weight:", neighbor_weights[node])
        print("edges weight:", edge_weights[mask[node]])
        print("probs:", prob_links[mask[node]])
        print("receivers:", receivers[mask[node]])
        print("from", last_node)
        print("to", target)
        print("next", next_node)
        print("local cost", local_cost)
        print("local hops", local_hops)
        print()
        if next_node == target:
            return local_cost + node_edge_weights[next_node_idx], local_hops + 1, False
        if next_node == last_node:
            neighbor_weights[node][next_node_idx] += 1
            return local_cost + node_edge_weights[next_node_idx], local_hops + 1, True

        cost, hops, is_reverse = flow(
            next_node,
            target,
            edge_weights,
            neighbor_weights,
            receivers,
            prob_links,
            mask,
            node,
        )
        local_hops = local_hops + hops + 1
        local_cost = local_cost + cost + node_edge_weights[next_node_idx]
        if is_reverse:
            neighbor_weights[node][next_node_idx] += 1
            data["weight"][next_node_idx] += 1
        else:
            return local_cost, local_hops, False


def reverse_link(graph, target, edge_weights, sources=None):
    prob_links = graph.edges
    receivers = graph.receivers
    senders = graph.senders
    if sources is None:
        sources = np.unique(senders)

    mask = mask_neighbors(senders)
    neighbor_weights = {}
    for node in sources:
        neighbor_weights[node] = np.zeros(mask[node].sum())
    print(neighbor_weights)

    metrics = {"cost": {}, "hops": {}}
    for node in sources:
        cost, hops, _ = flow(
            node, target, edge_weights, neighbor_weights, receivers, prob_links, mask
        )
        metrics["cost"][node] = cost
        metrics["hops"][node] = hops
        print("Finished for node {} with cost {} and hops {}".format(node, cost, hops))
        print()
    return metrics
