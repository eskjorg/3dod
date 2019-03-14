"""Main training script."""
import torch
from apex import amp

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL
from lib.postprocessing import PostProc
from lib.loss import LossHandler
from lib.model import Model
from lib.result import ResultSaver
from lib.utils import get_device, get_configs
from lib.visualize import Visualizer

from lib.data.loader import Loader

class Trainer():
    """Trainer."""

    def __init__(self, configs):
        """Constructor."""
        self._configs = configs
        self._data_loader = Loader((TRAIN, VAL), self._configs)
        self._result_saver = ResultSaver(configs)
        self._loss_handler = LossHandler(configs, self.__class__.__name__)
        self._checkpoint_handler = CheckpointHandler(configs)
        model = self._checkpoint_handler.init(Model(configs))
        optimizer = torch.optim.Adam(model.parameters(), lr=configs.training.learning_rate)
        self._model, self._optimizer = amp.initialize(model, optimizer, opt_level="O1")
        self._lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')
        self._post_proc = PostProc(configs)
        self._visualizer = Visualizer(configs)

    def train(self):
        """Main loop."""
        for epoch in range(1, self._configs.training.n_epochs + 1):
            self._run_epoch(epoch, TRAIN)
            val_score = -self._run_epoch(epoch, VAL)

            self._lr_scheduler.step(val_score)
            self._checkpoint_handler.save(self._model, epoch, val_score)

    def _run_epoch(self, epoch, mode):
        getattr(self._model, {TRAIN: 'train', VAL: 'eval'}[mode])()
        for batch_id, batch in enumerate(self._data_loader.gen_batches(mode)):
            outputs_cnn = self._run_model(batch.input, mode)
            loss = self._loss_handler.calc_loss(batch.gt_map, outputs_cnn)
            if mode == TRAIN:
                self._optimizer.zero_grad()
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
                self._optimizer.step()
            self._loss_handler.log_batch(epoch, batch_id, mode)
            results = self._post_proc.run(batch, outputs_cnn)
            self._result_saver.save(results, mode)

        score = self._result_saver.summarize_epoch(mode)
        if self._configs.data.dataformat == 'nuscenes':
            # TODO: implement evaluation for other datasets
            self._visualizer.report_score(epoch, score, mode)
        self._visualizer.report_loss(epoch, self._loss_handler.get_averages(), mode)
        self._visualizer.save_images(batch, results, mode, index=epoch)

        self._loss_handler.finish_epoch(epoch, mode)
        # TODO: implement evaluation for other datasets
        return score or sum(self._loss_handler.get_averages().values())

    def _run_model(self, inputs, mode):
        inputs = inputs.to(get_device(), non_blocking=True)
        with torch.set_grad_enabled(mode == TRAIN):
            return self._model(inputs)


def main(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, 'train')
    setup.prepare_environment()
    setup.save_settings(args)

    configs = get_configs(args.config_name)
    configs += vars(args)
    trainer = Trainer(configs)
    trainer.train()

if __name__ == '__main__':
    main(lib.setup)
