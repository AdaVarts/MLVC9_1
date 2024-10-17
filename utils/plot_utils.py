import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix


def plot_results_perceptron(weights, miss_rate, labels, pred):
    """Plot the results of the perceptron"""

    cm = confusion_matrix(labels, pred)
    acc = accuracy_score(labels, pred)

    fig, ax = plt.subplots(1, 3, figsize=(10, 3), dpi=200)
    ax[0].imshow(weights.reshape((16, 16)))
    ax[0].axis('off')
    ax[0].set_title("Learned weights")

    ax[1].plot(miss_rate)
    ax[1].set_title("Misclassifications")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Misclassification rate")

    df_cm = pd.DataFrame(cm, index = [i for i in ["Circle", "Square"]],
                  columns = [i for i in ["Circle", "Square"]])
    sns.heatmap(df_cm, annot=True, fmt='g', ax=ax[2])
    ax[2].set_title("Confusion matrix, accuracy: {:.2f}".format(acc))
    plt.tight_layout()
    plt.show()

# def plot_results_svm(classifier, labels, pred):
#     """Plot the results of the SVM"""
#     cm = confusion_matrix(labels, pred)
#     acc = accuracy_score(labels, pred)

#     fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

#     ax[0].imshow(classifier.kernel_G.cpu().numpy(), cmap='viridis')
#     ax[0].axis('off')
#     ax[0].set_title("Kernel matrix")

#     df_cm = pd.DataFrame(cm, index = [i for i in ["Circle", "Square"]],
#                   columns = [i for i in ["Circle", "Square"]])
#     sns.heatmap(df_cm, annot=True, fmt='g', ax=ax[1])
#     ax[1].set_title("Confusion matrix, accuracy: {:.2f}".format(acc))
#     plt.tight_layout()
#     plt.show()

def plot_results_svm(labels, pred_rbf, pred_linear):
    """Plot the results of the SVM"""
    cm_rbf = confusion_matrix(labels, pred_rbf)
    acc_rbf = accuracy_score(labels, pred_rbf)

    cm_linear = confusion_matrix(labels, pred_linear)
    acc_linear = accuracy_score(labels, pred_linear)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3), dpi=200)

    df_cm = pd.DataFrame(cm_linear, index = [i for i in ["Circle", "Square"]],
                  columns = [i for i in ["Circle", "Square"]])
    sns.heatmap(df_cm, annot=True, fmt='g', ax=ax[0])
    ax[0].set_title("Confusion matrix - linear, accuracy: {:.2f}".format(acc_linear))

    df_cm = pd.DataFrame(cm_rbf, index = [i for i in ["Circle", "Square"]],
                  columns = [i for i in ["Circle", "Square"]])
    sns.heatmap(df_cm, annot=True, fmt='g', ax=ax[1])
    ax[1].set_title("Confusion matrix - rbf, accuracy: {:.2f}".format(acc_rbf))
    plt.tight_layout()
    plt.show()

def vis_loss(loss_train_plot, loss_test_plot, acc_train_plot, acc_test_plot):
    """Visualize the loss and accuracy of the train and test set."""
    epochs_nr = np.arange(0, len(loss_train_plot), 1) + 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_nr, loss_train_plot, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(epochs_nr, loss_test_plot, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(which="both")
    # ax1.set_ylim([0, 1.01])
    ax1.legend(['Train', 'Test'], loc="lower left")

    # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    ax2.set_xlabel('Epochs')
    ax2.plot(epochs_nr, acc_train_plot, color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(which="both")
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    ax2.plot(epochs_nr, acc_test_plot, color=color, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim([0, 101])
    ax2.legend(['Train', 'Test'], loc="upper left")

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def vis_weights(model):
    input_weight = model.input_fc.weight.cpu().detach().numpy()
    hidden_weight = model.hidden_fc.weight.cpu().detach().numpy()
    output_weight = model.output_fc.weight.cpu().detach().numpy()

    print(input_weight.shape, hidden_weight.shape, output_weight.shape)

    fig, axs = plt.subplots(2, 5, figsize=(5, 2), dpi=300)

    for i, ax in enumerate(axs.reshape(-1)):
        ax.imshow(input_weight[i, :].reshape((16, 16)))
        ax.axis('off')
    fig.tight_layout()
    plt.show()

    fig = plt.figure()
    plt.imshow(hidden_weight)
    plt.axis('off')
    plt.show()

    fig = plt.figure()
    plt.imshow(output_weight)
    plt.axis('off')
    plt.show()

def vis_loss_and_weights(loss_train_plot, loss_test_plot, acc_train_plot, acc_test_plot, model):
    """Visualize the loss and accuracy of the train and test set, as well as the weights of the model."""
    epochs_nr = np.arange(0, len(loss_train_plot), 1) + 1
    #fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    fig = plt.figure(layout="constrained", figsize=(15, 10), dpi=100)
    subfigs = fig.subfigures(3, 1, wspace=0.07)
    axs0 = subfigs[0].subplots(1, 2)

    color = 'tab:red'
    axs0[0].set_xlabel('Epochs')
    axs0[0].set_ylabel('Loss', color=color)
    axs0[0].plot(epochs_nr, loss_train_plot, color=color)
    axs0[0].tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    axs0[0].set_ylabel('Loss', color=color)
    axs0[0].plot(epochs_nr, loss_test_plot, color=color, linestyle='dashed')
    axs0[0].tick_params(axis='y', labelcolor=color)
    axs0[0].grid(which="both")
    axs0[0].legend(['Train', 'Test'], loc="upper right")
    axs0[0].set_title("Loss")

    color = 'tab:red'
    axs0[1].set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    axs0[1].set_xlabel('Epochs')
    axs0[1].plot(epochs_nr, acc_train_plot, color=color)
    axs0[1].tick_params(axis='y', labelcolor=color)
    axs0[1].grid(which="both")
    color = 'tab:blue'
    axs0[1].set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    axs0[1].plot(epochs_nr, acc_test_plot, color=color, linestyle='dashed')
    axs0[1].tick_params(axis='y', labelcolor=color)
    axs0[1].set_ylim([0, 101])
    axs0[1].legend(['Train', 'Test'], loc="lower right")
    axs0[1].set_title("Accuracy")

    input_weight = model.input_fc.weight.cpu().detach().numpy()
    hidden_weight = model.hidden_fc.weight.cpu().detach().numpy()
    output_weight = model.output_fc.weight.cpu().detach().numpy()

    axs1 = subfigs[1].subplots(2, 5)

    subfigs[1].suptitle('Input Weights')
    for i, ax in enumerate(axs1.reshape(-1)):
        ax.imshow(input_weight[i, :].reshape((16, 16)))
        ax.axis('off')

    axs2 = subfigs[2].subplots(1, 2)

    axs2[0].imshow(hidden_weight)
    axs2[0].axis('off')
    axs2[0].set_title("Hidden Weights")

    axs2[1].imshow(output_weight)
    axs2[1].axis('off')
    axs2[1].set_title("Output Weights")

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

def plot_results_mlp(model):
    """Visualize the loss and accuracy of the train and test set, as well as the weights of the model."""
    epochs_nr = np.arange(0, len(model.loss_train_plot), 1) + 1
    #fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=100)
    fig = plt.figure(layout="constrained", figsize=(15, 10), dpi=100)
    subfigs = fig.subfigures(3, 1, wspace=0.07)
    axs0 = subfigs[0].subplots(1, 2)

    color = 'tab:red'
    axs0[0].set_xlabel('Epochs')
    axs0[0].set_ylabel('Loss', color=color)
    axs0[0].plot(epochs_nr, model.loss_train_plot, color=color)
    axs0[0].tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    axs0[0].set_ylabel('Loss', color=color)
    axs0[0].plot(epochs_nr, model.loss_test_plot, color=color, linestyle='dashed')
    axs0[0].tick_params(axis='y', labelcolor=color)
    axs0[0].grid(which="both")
    axs0[0].legend(['Train', 'Test'], loc="upper right")
    axs0[0].set_title("Loss")

    color = 'tab:red'
    axs0[1].set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    axs0[1].set_xlabel('Epochs')
    axs0[1].plot(epochs_nr, np.array(model.acc_train_plot)*100, color=color)
    axs0[1].tick_params(axis='y', labelcolor=color)
    axs0[1].grid(which="both")
    color = 'tab:blue'
    axs0[1].set_ylabel('Accuracy',
                   color=color)  # we already handled the x-label with ax1
    axs0[1].plot(epochs_nr, np.array(model.acc_test_plot)*100, color=color, linestyle='dashed')
    axs0[1].tick_params(axis='y', labelcolor=color)
    axs0[1].set_ylim([0, 101])
    axs0[1].legend(['Train', 'Test'], loc="lower right")
    axs0[1].set_title("Accuracy")

    input_weight = model.hidden_weight.T
    hidden_weight = model.hidden_weight2.T
    output_weight = model.output_weight.T

    axs1 = subfigs[1].subplots(2, 5)

    subfigs[1].suptitle('Input Weights')
    for i, ax in enumerate(axs1.reshape(-1)):
        ax.imshow(input_weight[i, :].reshape((16, 16)))
        ax.axis('off')

    axs2 = subfigs[2].subplots(1, 2)

    axs2[0].imshow(hidden_weight)
    axs2[0].axis('off')
    axs2[0].set_title("Hidden Weights")

    axs2[1].imshow(output_weight)
    axs2[1].axis('off')
    axs2[1].set_title("Output Weights")

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()

