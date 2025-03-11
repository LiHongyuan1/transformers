from pathlib import Path
import torch.multiprocessing as mp

import logging
import pytorch_lightning as pl
import hydra

from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from cross_view_transformer.common import setup_config, setup_experiment, load_backbone
from cross_view_transformer.callbacks.gitdiff_callback import GitDiffCallback
from cross_view_transformer.callbacks.visualization_callback import VisualizationCallback


log = logging.getLogger(__name__)

CONFIG_PATH = Path.cwd() / 'config'
CONFIG_NAME = 'config.yaml'


#lhy modify
def maybe_resume_training(experiment):
    if not experiment.resume:
        log.info('Resume training is disabled in the configuration.')
        return None

    save_dir = Path(experiment.save_dir).resolve()
    checkpoints = list(save_dir.glob(f'**/{experiment.uuid}/checkpoints/*.ckpt'))

    log.info(f'Searching {save_dir} for checkpoints.')

    if not checkpoints:
        log.info('No checkpoints found.')
        return None

    # 按修改时间排序，选择最新的检查点
    checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime)
    latest_checkpoint = checkpoints[-1]

    log.info(f'Found checkpoint: {latest_checkpoint}.')

    return str(latest_checkpoint)

#lhy modify
@hydra.main(config_path=str(CONFIG_PATH), config_name=CONFIG_NAME, version_base=None)

def main(cfg):
    setup_config(cfg)

    pl.seed_everything(cfg.experiment.seed, workers=True)

    Path(cfg.experiment.save_dir).mkdir(exist_ok=True, parents=False)

    # Create and load model/data
    model_module, data_module, viz_fn = setup_experiment(cfg)

    # Optionally load model
    ckpt_path = maybe_resume_training(cfg.experiment)

    #lhy modify
    ## 移除手动加载 backbone 的代码
    # if ckpt_path is not None:
    #     model_module.backbone = load_backbone(ckpt_path)

    # Loggers and callbacks
    logger = pl.loggers.WandbLogger(project=cfg.experiment.project,
                                    save_dir=cfg.experiment.save_dir,
                                    id=cfg.experiment.uuid)

   #lhy modify
    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(filename='model',
                        save_on_train_epoch_end=True,
                        every_n_epochs=1),

        VisualizationCallback(viz_fn, cfg.experiment.log_image_interval),
        GitDiffCallback(cfg)
    ]

    # Train
    trainer = pl.Trainer(logger=logger,
                         callbacks=callbacks,
                         #lhy modify
                         **cfg.trainer)
    trainer.fit(model_module, datamodule=data_module, ckpt_path=ckpt_path)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
