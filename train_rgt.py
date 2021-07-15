import hydra
from omegaconf import DictConfig, OmegaConf

from rgt.training import EstimatorRGT
from rgt.model import RoutingGraphTransformer


@hydra.main(config_path='configs', config_name='rgt')
def my_app(cfg: DictConfig) -> None:
    model = RoutingGraphTransformer(**cfg.model)
    estimator = EstimatorRGT(model, **cfg.train)
    print(estimator)

if __name__ == "__main__":
    my_app()
