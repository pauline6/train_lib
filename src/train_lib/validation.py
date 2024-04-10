from typing_extensions import Optional
from dataclasses import dataclass

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric
from torch.utils.data import DataLoader


@dataclass
class Evaluator:
    name: str
    engine: Engine
    loader: DataLoader
    event: Events
    metric_ckpt: Optional[ModelCheckpoint]


def validation_evaluator(name: str, val_engine: Engine, val_loader: DataLoader, event: Events,
                         val_metrics: Optional[dict[str, Metric]] = None, metric_checkpoint: Optional[str] = None):
    evaluator = Evaluator(name=name, engine=val_engine, loader=val_loader, event=event, metric_ckpt=metric_checkpoint)

    if val_metrics is not None:
        for name, metric in val_metrics.items():
            metric.attach(val_engine, name)

    return evaluator