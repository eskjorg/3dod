"""Main testing script."""
import logging
import torch

from apex import amp
amp_handle = amp.init()

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TEST
from lib.postprocessing import PostProc
from lib.model import Model
from lib.result import ResultSaver
from lib.utils import get_device, get_configs
from lib.visualize import Visualizer

from lib.data.loader import Loader

class Tester():
    """Tester."""
    def __init__(self, configs):
        """Constructor."""
        self._configs = configs
        self._data_loader = Loader((TEST,), self._configs)
        self._result_saver = ResultSaver(configs)
        self._checkpoint_handler = CheckpointHandler(configs)
        self._model = self._checkpoint_handler.init(Model(configs), force_load=True)
        self._post_proc = PostProc(configs)
        self._visualizer = Visualizer(configs)
        self._logger = logging.getLogger(self.__class__.__name__)

    def test(self):
        self._model.eval()
        cnt = 0
        for batch_id, batch in enumerate(self._data_loader.gen_batches(TEST)):
            if self._configs.loading[TEST]['max_nbr_batches'] is not None and batch_id >= self._configs.loading[TEST]['max_nbr_batches']:
                break
            outputs_cnn = self._run_model(batch.input)
            results = self._post_proc.run(batch, outputs_cnn)
            self._result_saver.save(results, TEST, batch)
            for sample_idx in range(len(batch.id)):
                self._visualizer.save_images(batch, outputs_cnn, results, TEST, index=cnt, sample=sample_idx)
                cnt += 1
            self._logger.info('Inference done for Batch {id:<5d}'.format(id=batch_id))
        self._result_saver.summarize_epoch(TEST)

    def _run_model(self, inputs):
        inputs = inputs.to(get_device(), non_blocking=True)
        with torch.no_grad():
            return self._model(inputs)

def main(setup):
    args = setup.parse_arguments()
    setup.setup_logging(args.experiment_path, 'test')
    setup.prepare_environment()
    setup.save_settings(args)

    configs = get_configs(args.config_name)
    configs += vars(args)
    if args.train_seqs is not None:
        configs['data']['sequences']['train'] = args.train_seqs.split(',')
    if args.group_labels is not None:
        configs['data']['group_labels'] = args.group_labels.split(',')
    tester = Tester(configs)
    tester.test()

if __name__ == '__main__':
    main(lib.setup)
