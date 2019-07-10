"""Main evaluation script."""
import logging
import torch

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TRAIN, VAL, TEST
from lib.postprocessing import PostProc
from lib.keypointrcnn.loss import LossHandler
# from lib.model import Model
from lib.keypointrcnn.model import Model
from lib.result import ResultSaver
from lib.utils import get_device, get_configs
from lib.visualize import Visualizer

from lib.keypointrcnn.loader import Loader

class Evaluator():
    """Evaluator."""
    def __init__(self, configs):
        """Constructor."""
        self._configs = configs
        self._data_loader = Loader((TRAIN, VAL, TEST), self._configs)
        self._result_saver = ResultSaver(configs)
        self._loss_handler = LossHandler(configs, self.__class__.__name__)
        self._checkpoint_handler = CheckpointHandler(configs)
        self._model = self._checkpoint_handler.init(Model(configs), force_load=True)
        self._model.eval()
        self._post_proc = PostProc(configs)
        self._visualizer = Visualizer(configs)
        self._logger = logging.getLogger(self.__class__.__name__)

    def eval(self):
        for mode in self._configs.eval_mode:
            self._model.eval()
            cnt = 0
            for batch_id, batch in enumerate(self._data_loader.gen_batches(mode)):
                outputs_cnn = self._run_model(batch.input, batch.targets, mode)

                if mode in (TRAIN, VAL):
                    # loss = self._loss_handler.calc_loss(outputs_cnn) # Not possible during eval...
                    self._loss_handler.log_batch(0, batch_id, mode)

                results = self._post_proc.run(batch, outputs_cnn)
                self._result_saver.save(results, mode, batch)
                for sample_idx in range(len(batch.id)):
                    self._visualizer.save_images(batch, outputs_cnn, results, mode, index=cnt, sample=sample_idx)
                    cnt += 1
                self._logger.info('Inference done for Batch {id:<5d}'.format(id=batch_id))
                if self._configs.loading[mode]['max_nbr_batches'] is not None and batch_id+1 >= self._configs.loading[mode]['max_nbr_batches']:
                    break
            self._result_saver.summarize_epoch(mode)

    def _run_model(self, inputs, targets, mode):
        #inputs = inputs.to(get_device(), non_blocking=True)
        inputs = [data.contiguous().to(get_device(), non_blocking=True) for data in inputs]
        targets = [{k: v.to(get_device()) for k, v in t.items()} for t in targets]
        with torch.set_grad_enabled(mode == TRAIN):
            return self._model(inputs, targets)
    # def _run_model(self, inputs):
    #     inputs = inputs.to(get_device(), non_blocking=True)
    #     with torch.no_grad():
    #         return self._model(inputs)

def main(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, 'val')
    setup.prepare_environment()
    setup.save_settings(args)

    configs = get_configs(args.config_name)
    configs += vars(args)

    # NOTE: The loading options for TEST is used also for TRAIN & VAL during evaluation.
    configs['loading'][TRAIN]['shuffle'] = configs['loading'][TEST]['shuffle']
    configs['loading'][VAL]['shuffle'] = configs['loading'][TEST]['shuffle']
    configs['loading'][TRAIN]['max_nbr_batches'] = configs['loading'][TEST]['max_nbr_batches']
    configs['loading'][VAL]['max_nbr_batches'] = configs['loading'][TEST]['max_nbr_batches']

    configs['data']['sequences']['train_extra'] = []
    if args.train_seqs is not None:
        configs['data']['sequences']['train'] = args.train_seqs.split(',')
    if args.group_labels is not None:
        configs['data']['class_labels'] = args.group_labels.split(',')
    evaluator = Evaluator(configs)
    configs['data']['data_loader'] = evaluator._data_loader
    evaluator.eval()

if __name__ == '__main__':
    main(lib.setup)
