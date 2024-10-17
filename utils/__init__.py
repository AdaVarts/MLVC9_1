from .convert_dataset_to_pytorch import \
    convert_dataset_to_pytorch as Dataset_PyTorch
from .generate_dataset import make_dataset as Dataset
from .mlp_pytorch import MLP, evaluate, train
from .plot_utils import (plot_results_mlp, plot_results_perceptron,
                         plot_results_svm, vis_loss, vis_loss_and_weights,
                         vis_weights)
