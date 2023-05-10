import torch
from utils.status import progress_bar
import numpy as np
from utils.training import mask_classes
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader


def mask_classes(outputs, dataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(self, dataset, last=False, eval_model='net'):
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :param eval_model: name of the model to evaluate
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    eval_model = getattr(self, eval_model)
    status = eval_model.training
    eval_model.eval()
    accs, accs_mask_classes = [], []
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        for data in test_loader:
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                if 'class-il' not in self.COMPATIBILITY:
                    outputs = eval_model(inputs, k)
                else:
                    outputs = eval_model(inputs)

                _, pred = torch.max(outputs.data, 1)
                correct += torch.sum(pred == labels).item()
                total += labels.shape[0]

                if dataset.SETTING == 'class-il':
                    mask_classes(outputs, dataset, k)
                    _, pred = torch.max(outputs.data, 1)
                    correct_mask_classes += torch.sum(pred == labels).item()

        print(f'Task {k} Accuracy: {correct / total * 100}')
        accs.append(correct / total * 100
                    if 'class-il' in self.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)

    eval_model.train(status)
    return accs, accs_mask_classes