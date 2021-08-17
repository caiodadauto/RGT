import hydra
from omegaconf import DictConfig, OmegaConf

from rgt.training import EstimatorRGT
from rgt.model import RoutingGraphTransformer


@hydra.main(config_path="configs", config_name="rgt")
def my_app(cfg: DictConfig) -> None:
    model = RoutingGraphTransformer(**cfg.model)

    # gen = init_generator(
    #     cfg.train.tr_path_data, 10, True, np.random.RandomState(), "gpickle", size=80
    # )
    # in_graphs, _, _ = next(gen)
    # targets = in_graphs.globals
    # print(model(in_graphs, targets, True))

    estimator = EstimatorRGT(model, **cfg.train)
    estimator.train()


if __name__ == "__main__":
    my_app()
