from tqdm import tqdm

import pytop
from gn_contrib.utils import networkx_to_graph_tuple_generator


__all__ = ["init_generator"]


def init_generator(
    path, n_batch, scaler, random_state, file_ext, seen_graphs=0, size=None, input_fields=None
):
    if size is not None:
        batch_bar = tqdm(
            total=size,
            initial=seen_graphs,
            desc="Processed Graphs",
            leave=False,
        )
    generator = networkx_to_graph_tuple_generator(
        pytop.batch_files_generator(
            path,
            file_ext,
            n_batch,
            dataset_size=size,
            shuffle=True,
            bidim_solution=False,
            input_fields=input_fields,
            random_state=random_state,
            seen_graphs=seen_graphs,
            scaler=scaler,
        )
    )
    if size is not None:
        return batch_bar, generator
    else:
        return None, generator
