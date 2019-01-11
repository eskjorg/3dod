"""Create batches."""

from collections import namedtuple

#from lib.data.loader import Batch

def collate_batch(batch_list):
    # TODO:
    pass
    # def collate_items(key):
    #
    # {key: [sample[key] for sample in batch_list] for key in batch_list[0]}
    # {key: collate_items(key) for key in batch_list[0]}
    # return Batch(annotation)

class PreProcessIf:
    """Pre processing interface."""
    def __init__(self, arg):
        self.arg = arg
        # TODO:

class Train(PreProcessIf):
    """docstring for Train."""
    def __init__(self, arg):
        super(Train, self).__init__()
        self.arg = arg
        # TODO:

class Validate(PreProcessIf):
    """docstring for Validate."""
    def __init__(self, arg):
        super(Validate, self).__init__()
        self.arg = arg
        # TODO:

class Test(PreProcessIf):
    """docstring for Test."""
    def __init__(self, arg):
        super(Test, self).__init__()
        self.arg = arg
        # TODO:
