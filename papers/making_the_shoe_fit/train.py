from functools import partial
import hydra
import copy
import jax
import wandb
from tqdm import tqdm

from src import training_utils as utils
from src.hydra_utils import get_dataset, get_model, get_optimizer

def initialize_logger(cfg):
    run = wandb.init(
        project="making_the_shoe_fit", 
        entity="shpotes", 
        name=f"{cfg.training.run_name} - {cfg.dataset.name}"
    )
    wandb.config = dict(cfg)
    return run

def train_and_evaluate(cfg):
    train_ds, test_ds = get_dataset(cfg)
    model = get_model(cfg)
    tx = get_optimizer(cfg)
    rng = jax.random.PRNGKey(42)

    state = utils.create_train_state(model, tx, rng)

    train_step = partial(utils.train_step, cfg.training.dp, model)
    test_step = partial(utils.test_step, model)

    for epoch in tqdm(range(1, cfg.training.num_epoch + 1)):
        for batch in train_ds:
            state, training_metrics = train_step(state, batch)
            
            wandb.log(training_metrics)
            
        test_metrics = test_step(state.params, test_ds)
        wandb.log(test_metrics)

    return state, test_metrics

@hydra.main(config_name="batch_lr_experiment", config_path="conf")
def main(cfg):
    for experiment in cfg.training:
        curr_cfg = copy.deepcopy(cfg)
        curr_cfg.training = cfg.training[experiment]
        
        run = initialize_logger(curr_cfg)
        train_and_evaluate(curr_cfg)
        run.finish()


if __name__ == '__main__':
    main()