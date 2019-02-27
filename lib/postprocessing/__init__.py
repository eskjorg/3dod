"""2D and 3D Detection from network outputs."""
from abc import abstractmethod
from collections import defaultdict
from importlib import import_module

class PostProc:
    def __init__(self, configs):
        self._modules = [getattr(import_module('.' + method, __name__), 'Runner')(configs)
                         for method in configs.postprocessing.methods]

    def run(self, batch, outputs):
        """Run selected post_proc modules on all frames."""
        batch_results = defaultdict(dict)
        for frame_index, frame_id in enumerate(batch.id):
            frame_results = outputs
            for module in self._modules:
                frame_results = module.run(frame_results, batch, frame_index)
            batch_results.update({frame_id: frame_results})
        return batch_results


class RunnerIf:
    """PostProc runner interface."""
    def __init__(self, configs, runner_configs):
        self._configs = configs
        self._runner_configs = getattr(configs.postprocessing, runner_configs)

    @abstractmethod
    def run(self, outputs, batch, frame_index):
        """Run post_proc for frame"""
