from datasets.seq_cifar10 import SequentialCIFAR10
from datasets.seq_cifar100 import SequentialCIFAR100
from datasets.seq_tinyimagenet import SequentialTinyImagenet
from datasets.utils.continual_dataset import ContinualDataset
from argparse import Namespace
from datasets.gcil_cifar100 import GCILCIFAR100

NAMES = {
    SequentialCIFAR10.NAME: SequentialCIFAR10,
    SequentialCIFAR100.NAME: SequentialCIFAR100,
    SequentialTinyImagenet.NAME: SequentialTinyImagenet,
    GCILCIFAR100.NAME: GCILCIFAR100,
}

def get_dataset(args: Namespace) -> ContinualDataset:
    """
    Creates and returns a continual dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in NAMES.keys()
    return NAMES[args.dataset](args)


def get_gcl_dataset(args: Namespace):
    """
    Creates and returns a GCL dataset.
    :param args: the arguments which contains the hyperparameters
    :return: the continual dataset
    """
    assert args.dataset in GCL_NAMES.keys()
    return GCL_NAMES[args.dataset](args)
