"""Evaluator."""


class Evaluator:
    """Evaluator."""
    def __init__(self, configs):
        self._configs = configs
        self._results = {}

    def calc_batch(self, detections, annotations):
        # TODO: Implement evaluation
        return None
        for metric in self._configs.abcd:
            for det, anno in zip(detections.values(), annotations):
                n_gt = 0
                pass

    def summarize_epoch(self):
        # TODO: Implement evaluation
        return 0
