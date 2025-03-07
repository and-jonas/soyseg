import hydra
from flash.image import SemanticSegmentation, SemanticSegmentationData
from transforms import set_input_transform_options
from config import SoySegConfig
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from flash import Trainer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch
import torchmetrics
import gc
import BatchSizeFinder


metrics = [
    torchmetrics.Accuracy(num_classes=3, mdmc_reduce='global', multiclass=True),
    torchmetrics.F1Score(num_classes=3, mdmc_reduce='global', multiclass=True),
    torchmetrics.Precision(num_classes=3, mdmc_reduce='global',  multiclass=True),
    torchmetrics.Recall(num_classes=3, mdmc_reduce='global', multiclass=True),
    ]


@hydra.main(config_path="conf", config_name="config_res_seg", version_base=None)
def objective(cfg: SoySegConfig):
    print(cfg)

    batch_size = BatchSizeFinder.find_max_batch_size_simple(
        backbone=cfg.model.backbone,
        head=cfg.model.head,
        size=cfg.transform.size,
        crop_factor=1,
        max_batch_size=400,
    )

    # Data Transformations
    trf = set_input_transform_options(
        head=cfg.model.head,
        size=cfg.transform.size,
        crop_factor=cfg.transform.crop_factor,
        blur_kernel_size=cfg.transform.blur_kernel_size,
        p_color_jitter=cfg.transform.p_color_jitter,
        rand_rot=cfg.transform.rand_rot,
        scaling=cfg.transform.scaling,
    )

    datamodule = SemanticSegmentationData.from_folders(
        train_folder=cfg.paths.train_folder,
        train_target_folder=cfg.paths.train_target_folder,
        val_folder=cfg.paths.val_folder,
        val_target_folder=cfg.paths.val_target_folder,
        train_transform=trf,
        val_transform=trf,
        test_transform=trf,
        predict_transform=trf,
        num_classes=3,
        batch_size=batch_size,
        num_workers=cfg.train.num_workers,
    )

    # print("Train dataset size:", len(datamodule.train_dataloader()))
    # print("Validation dataset size:", len(datamodule.val_dataloader()))

    pretrained = False if cfg.train.strategy == "train" else True

    # Build the model
    model = SemanticSegmentation(
        pretrained=pretrained,
        backbone=cfg.model.backbone,
        head=cfg.model.head,
        metrics=metrics,
        num_classes=datamodule.num_classes,
        learning_rate=cfg.train.learning_rate,
        optimizer=cfg.train.optimizer,
    )

    early_stopping = EarlyStopping(monitor='val_f1score', mode='max', patience=5)
    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)

    # must be specified inside the objective()
    # overwrites otherwise
    logger = TensorBoardLogger(save_dir="/projects/SoySeg",
                               # default_hp_metric=True,
                               name="soyseg")

    logger.log_hyperparams({"head": cfg.model.head,
                            "backbone": cfg.model.backbone,
                            "strategy": cfg.train.strategy,
                            "blur_kernel_size": cfg.transform.blur_kernel_size,
                            "optimizer": cfg.train.optimizer,
                            "learning_rate": cfg.train.learning_rate,
                            # "momentum": momentum,
                            "size": cfg.transform.size,
                            "rand_rot": cfg.transform.rand_rot,
                            "scaling": cfg.transform.scaling,
                            "p_color_jitter": cfg.transform.p_color_jitter,
                            "batch_size": datamodule.batch_size})

    # 3. Create the trainer and finetune the model
    trainer = Trainer(max_epochs=cfg.train.max_epochs,
                      move_metrics_to_cpu=False,
                      gpus=[cfg.train.gpu],
                      precision=16,
                      logger=logger,
                      callbacks=[early_stopping, lr_monitor],
                      enable_checkpointing=False
                      )

    # Train the model for Optuna to understand the current
    # Hyperparameter combination's behaviour.
    if cfg.train.strategy == "train":
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.finetune(model, datamodule=datamodule, strategy=cfg.train.strategy)
    # trainer.save_checkpoint("/projects/SoySeg/soyseg_ff.pt")

    # The extra step to tell Optuna which value to base the
    # optimization routine on.
    # But this only gets the metrics of the LAST iteration
    value = trainer.callback_metrics["val_f1score"].item()
    # Get the highest metric from all iterations; based on
    # https://www.programcreek.com/python/example/114903/tensorboard.backend.event_processing.event_accumulator.EventAccumulator
    event_acc = EventAccumulator(path=logger.log_dir)
    event_acc.Reload()
    tags = event_acc.Tags()['scalars']  # available metrics
    v = event_acc.Scalars('val_f1score')  # use validation f1score
    v = [v[i].value for i in range(len(v))]
    value = max(v)  # get the highest value, since direction='maximize'

    # log this value
    logger.log_metrics({"hp_metric": value})

    # get rid of everything
    del datamodule, model, trainer, trf, logger
    gc.collect()
    torch.cuda.empty_cache()

    return value


if __name__ == "__main__":
    objective()