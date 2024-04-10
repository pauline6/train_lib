from typing import Callable, Union
from dataclasses import dataclass

from ignite.engine import Engine, Events
from torch.utils.data import DataLoader
from ignite.contrib.handlers import TensorboardLogger


@dataclass
class Inferencer:
    name: str
    type: str = ["image" | "figure" ]
    engine: Engine
    loader: DataLoader
    event: Events
    output_transform: Callable


def inference_configuration(name: str, engine: Engine, loader: DataLoader, event: Events, 
                            output_transform: Callable, type: str = ["image" | "figure" ]) -> Inferencer:
    infercer = Inferencer(name=name, type=type, engine=engine, loader=loader, event=event,
                          output_transform=output_transform)
    return infercer