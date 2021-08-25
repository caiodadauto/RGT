import os
import joblib
import tempfile

import mlflow as mlf
from omegaconf import DictConfig, ListConfig


__all__ = ["set_mlflow_env", "log_params_from_omegaconf_dict"]


def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)


def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f"{parent_name}.{k}", v)
            else:
                mlf.log_param(f"{parent_name}.{k}", v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlf.log_param(f"{parent_name}.{i}", v)


def save_pickle(name, obj):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            joblib.dump(obj, f)
        mlf.log_artifact(path)

def load_pickle(name, run):
    artifact_path = os.path.normpath(run.info.artifact_uri).split(":")[-1]
    path = os.path.join(artifact_path, f"{name}.pkl")
    return joblib.load(path)
