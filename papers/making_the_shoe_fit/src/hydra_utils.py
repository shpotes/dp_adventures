from functools import partial
import flax
from hydra.utils import instantiate, get_method, get_class
from omegaconf import OmegaConf
from tensorflow_datasets import Split
from src.dataset import load_dataset

def get_dataset(cfg):
    train_ds = load_dataset(
        cfg.dataset.name,
        cfg.dataset.num_labels,
        cfg.dataset.batch_size
    )
    train_ds = list(train_ds.as_numpy_iterator())
    test_ds = load_dataset(
        cfg.dataset.name,
        cfg.dataset.num_labels,
        cfg.dataset.batch_size,
        split=Split.TEST,
        shuffle=False
    )
    test_ds = next(
        test_ds
        .unbatch()
        .batch(60_000)
        .as_numpy_iterator()
    )

    return train_ds, test_ds

def get_model(cfg):
    return instantiate(cfg.model)


def get_optimizer(cfg):
    # TODO: add dp support
    return instantiate(cfg.training.optimizer)