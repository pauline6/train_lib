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


def validation_evaluator(name: str, engine: Engine, loader: DataLoader, event: Events,
                         metrics: Optional[dict[str, Metric]] = None, metric_checkpoint: Optional[str] = None):
    evaluator = Evaluator(name=name, engine=engine, loader=loader, event=event, metric_ckpt=metric_checkpoint)

    if metrics is not None:
        for name, metric in metrics.items():
            metric.attach(engine, name)

    return evaluator