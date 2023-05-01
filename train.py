import torch
import torch.nn as nn
import yaml
from model import *
from dataset import *
from my_utils import *
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from label_list import all_label_list
import shutil
import pdb


def load_model(general_cfg, model_cfg, n_classes, ckpt_path=None):
    model_type = general_cfg['options']['model_type']
    if model_type == 'rgcn':
        if ckpt_path is not None:
            model = RGCN_Model.load_from_checkpoint(
                        checkpoint_path=ckpt_path,
                        general_config=general_cfg, 
                        model_config=model_cfg, 
                        n_classes=n_classes
                    )
        else:
            model = RGCN_Model(
                general_config=general_cfg, 
                model_config=model_cfg, 
                n_classes=n_classes
            )
    else:
        raise ValueError(f'Model type {model_type} is not supported yet')    
    
    return model


def train():
    # load config and setup
    with open("configs/train_cfg.yaml") as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)

    model_type = general_cfg['options']['model_type']
    with open(os.path.join('configs', model_type+'.yaml')) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    experiment_dir = get_experiment_dir(general_cfg['training']['ckpt_save_dir'])
    os.makedirs(experiment_dir, exist_ok=True)
    shutil.copy("configs/train_cfg.yaml", experiment_dir)
    shutil.copyfile(os.path.join('configs', model_type+'.yaml'), os.path.join(experiment_dir, 'model_cfg.yaml'))

    # get data
    data_module = GraphDataModule(config=general_cfg)
    
    # init model
    label_list = all_label_list[general_cfg['data']['label_list']]
    model = load_model(
        general_cfg, 
        model_cfg, 
        n_classes=len(label_list), 
        ckpt_path=None
    )

    # callbacks
    model_ckpt = ModelCheckpoint(
        monitor='val_f1',
        mode='max',
        dirpath=experiment_dir,
        filename='model-{epoch:02d}-{train_loss:.3f}-{val_loss:.3f}-{val_f1:.3f}',
        save_top_k=3,
        auto_insert_metric_name=True,
        every_n_epochs=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # tensorboard logger
    logger = TensorBoardLogger(
        save_dir=experiment_dir,
        name='',
        version=''
    )
    
    # trainer
    trainer = Trainer(
        accelerator='cpu', 
        devices=1,
        max_epochs=general_cfg['training']['num_epoch'],
        log_every_n_steps=3,
        auto_scale_batch_size=True,
        callbacks=[model_ckpt, lr_monitor],
        logger=logger,
        # overfit_batches=1,
        # fast_dev_run=True,
    )

    # train
    if general_cfg['training']['prev_ckpt_path'] is not None:
        trainer.fit(model=model, datamodule=data_module, ckpt_path=general_cfg['training']['prev_ckpt_path'])
    else:
        trainer.fit(model=model, datamodule=data_module)



if __name__ == '__main__':
    train()
