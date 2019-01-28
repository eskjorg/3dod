"""Main testing script."""
import torch

from apex import amp
amp_handle = amp.init()

import lib.setup
from lib.checkpoint import CheckpointHandler
from lib.constants import TEST
from lib.detection import Detector
from lib.evaluate import Evaluator
from lib.model import Model
from lib.save import ResultSaver
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
        self._detector = Detector(configs)
        self._evaluator = Evaluator(configs)
        self._visualizer = Visualizer(configs)

    def test(self):
        self._model.eval()
        for batch_id, batch in enumerate(self._data_loader.gen_batches(TEST)):
            outputs_cnn = self._run_model(batch.input)
            detections = self._detector.run_detection(batch, outputs_cnn)
            self._result_saver.save(detections, TEST)
            self._evaluator.calc_batch(detections, batch.annotation)
            self._visualizer.save_images(batch, detections, TEST, index=batch_id)
        score = self._evaluator.summarize_epoch()

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
    tester = Tester(configs)
    tester.test()

if __name__ == '__main__':
    main(lib.setup)
