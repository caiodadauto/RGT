import hydra
import mlflow as mlf
from omegaconf import DictConfig

from rgt.training import Train
from rgt.utils_mlf import set_mlflow

@hydra.main(config_path="configs", config_name="rgt")
def my_app(cfg: DictConfig) -> None:
    run = set_mlflow(
        cfg.mlflow.exp_name,
        cfg.mlflow.exp_tags,
        cfg.mlflow.run_tags,
        cfg.mlflow.run_id,
        cfg.mlflow.get_last_run,
    )
    with mlf.start_run(run_id=run.info.run_id):
        Train(cfg.model, cfg.estimator)


if __name__ == "__main__":
    my_app()
