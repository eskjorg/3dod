"""Main training script."""
import torch

from lib.setup import parse_arguments
from lib.setup import setup_logging
from lib.setup import prepare_environment
from lib.setup import save_settings
from lib.setup import get_configs

from lib.detection import run_detection
from lib.constants import TRAIN, VAL, CONFIG_PATH

class Trainer():
    """Trainer."""

    def __init__(self, settings, configs):
        """Constructor."""
        self._settings = settings
        self._configs = configs
        self._logger = None
        self._data_loader = None
        self._result_saver = None
        self._loss_handler = None
        self._checkpoint_handler = None
        self._model = None
        self._optimizer = torch.optim.Adam(self._model.parameters(),
                                           lr=configs.training.learning_rate)
        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max')
        self._visualizer = None

    def train(self):
        """Main loop."""
        for epoch in range(1, self._settings.n_epochs + 1):
            self._run_epoch(epoch, TRAIN)
            score_val = self._run_epoch(epoch, VAL)

            self._lr_scheduler.step(score_val)
            self._checkpoint_handler.save(self._model, epoch, score=score_val)

    def _run_epoch(self, epoch, mode):
        getattr(self._model, {TRAIN: 'train', VAL: 'eval'}[mode])()
        for batch_id, batch in self._data_loader.gen_batches(mode):
            outputs_cnn = self._run_model(batch.inputs, mode)
            detections = run_detection(outputs_cnn)

            self._result_saver.saver(detections, mode)
            loss, task_losses = self._loss_handler.calc_losses(batch, outputs_cnn, detections)
            if mode == TRAIN:
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            self._logger.log_batch(mode, epoch, batch_id, loss, task_losses)
            self._visualizer.save_images(epoch, batch, detections, mode)

        self._visualizer.report_loss(epoch, self._loss_handler.get_averages(), mode)
        score = self._calculate_score(epoch, mode)
        self._visualizer.report_score(epoch, score, mode)

        self._logger.finish_epoch(mode, epoch)  #self._log.epoch('Training epoch %s done!', epoch)
        return score


    def _run_model(self, inputs, mode):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = inputs.to(device)
        #inputs = inputs.float()
        with torch.set_grad_enabled(mode == TRAIN):
            return self._model(inputs)


def main():
    args = parse_arguments()
    setup_logging(args.experiment_path, 'train')
    prepare_environment(args)
    save_settings(args)

    trainer = Trainer(args, get_configs(CONFIG_PATH))
    trainer.train()

if __name__ == '__main__':
    main()
