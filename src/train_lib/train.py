from typing import Optional, Callable
from typing_extensions import Self

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Metric
from torch.utils.data import DataLoader
from torch.nn import Module
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
import numpy as np

from .validation import validation_evaluator
from .infer import inference_configuration

class Trainer:
    def __init__(self: Self,
                 train_engine: Engine,
                 train_loader: DataLoader,
                 model: Module,
                 tb_log_dir: str,
                 ckpt_dir: str = "checkpoint",
                 max_epochs: int = 100,
                 metric_ckpt: Optional[str] = None
                 ) -> None:
        """Training class

        Args:
            train_engine (Engine): Training engine.
            train_loader (DataLoader): Training DataLoader.
            model (Module): Training Model.
            tb_log_dir (str): Log directory for Tensorboard.
            max_epochs (int, optional): Maximal Number of epoch. Defaults to 100.
        """
        super(Trainer, self).__init__()
        self.train_engine = train_engine
        self.train_loader = train_loader
        self.max_epochs = max_epochs
        self.model = model
        self.evaluators = []
        self.inferencers = []
        self.losses = {}
        self.metric_ckpt = metric_ckpt
        self.ckpt_dir = ckpt_dir
        self.tb_logger = TensorboardLogger(log_dir=tb_log_dir)

    def add_validation(self,
                       name: str,
                       engine: Engine,
                       loader: DataLoader,
                       event: Events,
                       metrics: Optional[dict[str, Metric]] = None,
                       metric_checkpoint: Optional[str] = None):
        """Add a validation phase during the training.

        Args:
            name (str): Validation name.
            engine (Engine): Validation engine.
            loader (DataLoader): Validation DataLoader.
            event (Events): When the validation started in relation to the training engine timeline.
            metrics (Optional[dict[str, Metric]], optional): Metrics. Defaults to None.
            model_checkpoint (Optional[ModelCheckpoint], optional): Module to save model checkpoints. Defaults to None.
        """
        evaluator = validation_evaluator(name=name, engine=engine, loader=loader, event=event, metrics=metrics,
                                         metric_checkpoint=metric_checkpoint)
        self.evaluators.append(evaluator)
    
    def add_inference(self: Self,
                      name: str,
                      engine: Engine,
                      loader: DataLoader,
                      event: Events,
                      output_transform: Callable,
                      type_infer: str) -> None:
        infercer = inference_configuration(name=name, type_infer=type_infer, engine=engine, loader=loader,
                                           event=event, output_transform=output_transform)
        self.inferencers.append(infercer)

    def train(self):
        """Run the training with the validation previously added."""
        @self.train_engine.on(Events.EPOCH_STARTED)
        def log_train_start_epoch(engine: Engine):
            print(f"----- START Training Epoch {engine.state.epoch} -----")

        @self.train_engine.on(Events.ITERATION_COMPLETED)
        def log_training_loss(engine: Engine):
            for key, value in engine.state.output["loss"].items():
                print(f"Iter[{self.train_engine.state.iteration}]: {key}: {value:.2f}")
                if key not in self.losses.keys():
                    self.losses[key] = [value]
                else:
                    self.losses[key].append(value)

        @self.train_engine.on(Events.EPOCH_COMPLETED)
        def log_mean_training_loss(engine: Engine):
            print(f"=== END Training Loss Epoch[{engine.state.epoch}] ===")
            iteration = 1
            for key, value in self.losses.items():
                mean_loss = np.mean(np.array(value))
                print(f"{key}: {mean_loss:.2f}")
                key = "training/" + key
                self.tb_logger.writer.add_scalar(key, mean_loss)
            self.losses = {}

        for evaluator in self.evaluators:
            @evaluator.engine.on(Events.EPOCH_STARTED)
            def log_val_start_epoch(engine: Engine):
                print(f"----- START Validation Epoch {engine.state.epoch} -----")

            @self.train_engine.on(evaluator.event)
            def log_evaluator_results(engine: Engine):
                evaluator.engine.run(evaluator.loader)
                metrics = evaluator.engine.state.metrics
                if bool(metrics):
                    print(f"{evaluator.name} Results - Epoch[{self.train_engine.state.epoch}]")
                    for key, value in metrics:
                        print(f"{key}: {value:2f}")
                
            @evaluator.engine.on(Events.ITERATION_COMPLETED)
            def log_val_loss(engine: Engine):
                for key, value in engine.state.output["loss"].items():
                    print(f"Iter[{evaluator.engine.state.iteration}]: {key}: {value:.2f}")
                    if key not in self.losses.keys():
                        self.losses[key] = [value]
                    else:
                        self.losses[key].append(value)
            
            @evaluator.engine.on(Events.EPOCH_COMPLETED)
            def log_mean_val_loss(engine: Engine):
                print(f"=== END Validation Loss Epoch [{self.train_engine.state.epoch}] ===")
                for key, value in self.losses.items():
                    mean_loss = np.mean(np.array(value))
                    print(f"{key}: {mean_loss:.2f}")
                    key = evaluator.name + "/" + key
                    self.tb_logger.writer.add_scalar(key, mean_loss)
                self.losses = {}

            if evaluator.metric_ckpt is not None:
                def score_function(engine):
                    return -engine.state.output["loss"][evaluator.metric_ckpt]
                
                model_checkpoint = ModelCheckpoint(
                    self.ckpt_dir,
                    n_saved=2,
                    filename_prefix="ckpt",
                    score_function=score_function,
                    score_name=evaluator.metric_ckpt,
                    global_step_transform=global_step_from_engine(self.train_engine))

                evaluator.engine.add_event_handler(evaluator.event, model_checkpoint, {"model": self.model})

            
        if self.metric_ckpt is not None:
            def score_function(engine: Engine):
                return -engine.state.output["loss"][evaluator.metric_ckpt]

            model_checkpoint = ModelCheckpoint(
                self.ckpt_dir,
                n_saved=2,
                filename_prefix="ckpt",
                score_function=score_function,
                score_name=self.metric_ckpt,
                global_step_transform=global_step_from_engine(self.train_engine))

            self.train_engine.add_event_handler(Events.EPOCH_COMPLETED, model_checkpoint, {"model": self.model})
        
        for inferencer in self.inferencers:
            @self.train_engine.on(inferencer.event)
            def run_inference(engine: Engine):
                inferencer.engine.run(inferencer.loader)
                output = inferencer.engine.state.output
                figure = inferencer.output_transform(output)
                if inferencer.type_infer == "figure":
                    self.tb_logger.writer.add_figure(tag=inferencer.name, figure=figure)

        self.tb_logger.attach_output_handler(
            self.train_engine,
            event_name=Events.ITERATION_COMPLETED,
            tag="Loss training",
            output_transform=lambda loss: loss["loss"]
        )

        for evaluator in self.evaluators:
            self.tb_logger.attach_output_handler(
                evaluator.engine,
                event_name=evaluator.event,
                tag=evaluator.name,
                metric_names="all",
                global_step_transform=global_step_from_engine(self.train_engine),
            )

        self.train_engine.run(self.train_loader, self.max_epochs)

        self.tb_logger.close()
