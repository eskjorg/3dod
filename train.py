"""Main training script."""
import torch

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL, CONFIG_PATH
from lib.detection import Detector
from lib.evaluation import Evaluator
from lib.log import Logger
from lib.model import Model
from lib.utils import get_device

from lib.data.loader import Loader

class Trainer():
    """Trainer."""

    def __init__(self, settings, configs):
        """Constructor."""
        self._settings = settings
        self._configs = configs
        self._logger = Logger(self.__class__.__name__)
        self._data_loader = Loader((TRAIN, VAL), self._configs)
        self._result_saver = None  # TODO:
        self._loss_handler = None  # TODO:
        self._checkpoint_handler = CheckpointHandler(settings)
        self._model = self._checkpoint_handler.init(Model())
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=configs.training.learning_rate)
        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max')
        self._detector = None  # TODO:
        self._evaluator = None  # TODO:
        self._visualizer = None  # TODO:

    def train(self):
        """Main loop."""
        for epoch in range(1, self._settings.n_epochs + 1):
            self._run_epoch(epoch, TRAIN)
            val_score = self._run_epoch(epoch, VAL)

            self._lr_scheduler.step(val_score)
            self._checkpoint_handler.save(self._model, epoch, val_score)

    def _run_epoch(self, epoch, mode):
        getattr(self._model, {TRAIN: 'train', VAL: 'eval'}[mode])()
        for batch_id, batch in self._data_loader.gen_batches(mode):
            outputs_cnn = self._run_model(batch.inputs, mode)
            detections = self._detector.run_detection(outputs_cnn)

            self._result_saver.saver(detections, mode)
            loss, task_losses = self._loss_handler.calc_losses(batch, outputs_cnn, detections)
            if mode == TRAIN:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            self._logger.log_batch(epoch, batch_id, loss, task_losses, mode)
            self._visualizer.save_images(epoch, batch, detections, mode)

        self._visualizer.report_loss(epoch, self._loss_handler.get_averages(), mode)
        score = self._evaluator.calculate_score(epoch, mode)
        self._visualizer.report_score(epoch, score, mode)

        self._logger.finish_epoch(epoch, mode)
        return score


    def _run_model(self, inputs, mode):
        inputs = inputs.to(get_device())
        #inputs = inputs.float()
        with torch.set_grad_enabled(mode == TRAIN):
            return self._model(inputs)


def main(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, 'train')
    setup.prepare_environment(args)
    setup.save_settings(args)

    trainer = Trainer(args, setup.get_configs(CONFIG_PATH))
    trainer.train()

if __name__ == '__main__':
    main(lib.setup)
