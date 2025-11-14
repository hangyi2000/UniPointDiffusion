# ------------------------------------------------------------------------------------
# Enhancing Transformers
# Copyright (c) 2022 Thuan H. Nguyen. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import argparse
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.loggers import Logger, TensorBoardLogger

from typing import Tuple, List
import warnings

from pcld.utils import get_config_from_file, instantiate_from_config
from pcld.utils.callbacks.callback import SetupCallback

warnings.filterwarnings("ignore")


def setup_callbacks(exp_config: DictConfig, config: DictConfig) -> Tuple[List[Callback], Logger]:
    # now = datetime.now().strftime("%d%m%Y_%H%M%S")
    # basedir = Path(exp_config.output_dir, now)
    basedir = Path(exp_config.output_dir)
    os.makedirs(basedir, exist_ok=True)

    setup_callback = SetupCallback(config, exp_config, basedir)
    checkpoint_callback = ModelCheckpoint(
        dirpath=setup_callback.ckptdir,
        filename="model-ckpt-{epoch:02d}",
        monitor=exp_config.monitor,
        mode="max",
        save_top_k=-1,
        verbose=False,
        every_n_epochs=exp_config.every_n_epochs
    )

    logger = TensorBoardLogger(
        save_dir=str(setup_callback.logdir),
        name="tensorboard"
    )

    all_callbacks = [setup_callback, checkpoint_callback]

    if "logger" in config:
        custom_callback = instantiate_from_config(config.logger)
        all_callbacks.append(custom_callback)

    return all_callbacks, logger


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-nn", "--num_nodes", type=int, default=1)
    parser.add_argument("-ng", "--num_gpus", type=int, default=1)
    parser.add_argument("-u", "--update_every", type=int, default=1)
    parser.add_argument("-e", "--epochs", type=int, default=100)
    parser.add_argument("-lr", "--base_lr", type=float, default=4.5e-6)
    parser.add_argument("-a", "--use_amp", default=False, action="store_true")
    parser.add_argument("--gradient_clip_val", type=float, default=None)
    parser.add_argument("--gradient_clip_algorithm", type=str, default=None)
    parser.add_argument("--every_n_epochs", type=int, default=1)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--monitor", type=str, default="val/total_loss")
    parser.add_argument(
        "--scale_lr",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument("--output_dir", type=str, help="the output directory to save everything.")
    parser.add_argument("--ckpt_path", type=str, default="", help="the restore checkpoints.")
    args = parser.parse_args()

    # Set random seed
    pl.seed_everything(args.seed)

    # Load configuration
    config = get_config_from_file(args.config)

    exp_config = OmegaConf.create({"name": args.config,
                                   "epochs": args.epochs,
                                   "update_every": args.update_every,
                                   "base_lr": args.base_lr,
                                   "scale_lr": args.scale_lr,
                                   "use_amp": args.use_amp,
                                   "gradient_clip_val": args.gradient_clip_val,
                                   "gradient_clip_algorithm": args.gradient_clip_algorithm,
                                   "output_dir": args.output_dir,
                                   "every_n_epochs": args.every_n_epochs,
                                   "log_every_n_steps": args.log_every_n_steps,
                                   "monitor": args.monitor})

    # Setup callbacks
    callbacks, logger = setup_callbacks(exp_config, config)

    # Build data modules
    data: pl.LightningDataModule = instantiate_from_config(config.dataset)
    data.prepare_data()
    data.setup()

    # Build model
    model: pl.LightningModule = instantiate_from_config(config.model)
    base_lr = exp_config.base_lr
    nodes = args.num_nodes
    ngpus = args.num_gpus
    accumulate_grad_batches = exp_config.update_every
    batch_size = config.dataset.params.batch_size

    if args.scale_lr:
        model.learning_rate = accumulate_grad_batches * nodes * ngpus * batch_size * base_lr
        print(
            "Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (nodes) * {} (num_gpus) "
            "* {} (batchsize) * {:.2e} (base_lr)".format(model.learning_rate, accumulate_grad_batches,
                                                         nodes, ngpus, batch_size, base_lr))
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # Build trainer
    if args.num_nodes > 1 or args.num_gpus > 1:
        ddp_strategy = DDPStrategy(find_unused_parameters=False)
        # ddp_strategy = DDPStrategy(find_unused_parameters=True)
    else:
        ddp_strategy = None

    trainer = pl.Trainer(max_epochs=args.epochs,
                         precision=16 if args.use_amp else 32,
                         callbacks=callbacks,
                         accelerator="gpu",
                         devices=args.num_gpus,
                         num_nodes=args.num_nodes,
                         strategy=ddp_strategy,
                         gradient_clip_val=args.gradient_clip_val,
                         gradient_clip_algorithm=args.gradient_clip_algorithm,
                         accumulate_grad_batches=args.update_every,
                         logger=logger,
                         log_every_n_steps=args.log_every_n_steps)

    trainer.test(model, datamodule=data, ckpt_path=args.ckpt_path)

