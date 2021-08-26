import hydra
from omegaconf import DictConfig

from rgt.training import Train


@hydra.main(config_path="configs", config_name="rgt")
def my_app(cfg: DictConfig) -> None:
    Train(
        cfg.mlflow.exp_name,
        cfg.mlflow.exp_tags,
        cfg.mlflow.run_tags,
        cfg.mlflow.run_id,
        cfg.mlflow.get_last_run,
        cfg.model,
        cfg.estimator,
    )


if __name__ == "__main__":
    my_app()
