from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
from torch import logical_and as t_and, sum as t_sum, tensor as t

def display_loss(ax, history, label):
  ax.plot(history['ls'].numpy(), label=label)
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Loss')
  ax.legend(loc='upper right')

def display_acc(ax, history, label, num_of_classes = 4):
  y = history['ys']
  y_hat = history['y_hats']

  acc_history = t_sum(y == y_hat, dim = 1) / len(y[0])

  ax.plot(acc_history, label=label)
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Acc')
  ax.legend(loc='upper left')

def display_recall(ax, history, label, num_of_classes = 4):
  y = history['ys']
  y_hat = history['y_hats']

  acc_history = t_sum(y == y_hat, dim = 1) / len(y[0])

  ax.plot(acc_history, label=label)
  ax.set_xlabel('Epochs')
  ax.set_ylabel('Acc')
  ax.legend(loc='upper left')