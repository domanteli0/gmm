import torch.nn as nn
import torch

DIM = 64
NUM_CHANNELS = 3
NUM_CLASSES = 4

class Model(nn.Module):
  def __init__(self, num_classes = NUM_CLASSES):
    super(Model, self).__init__()
    self.softmax = nn.Softmax()

    self.conv1_1 = torch.nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=128, kernel_size=3, padding = 1)
    self.conv1_2 = torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 1)
    self.pool1 = torch.nn.MaxPool2d((2, 2), (2, 2))
    self.dropout1 = nn.Dropout2d(p=0.2)

    self.conv2_1 = torch.nn.Conv2d(128, 256, (3, 3), padding = 'same')
    self.conv2_2 = torch.nn.Conv2d(256, 256, (3, 3), padding = 'same')
    self.pool2 = torch.nn.MaxPool2d((2, 2), (2, 2))
    self.dropout2 = nn.Dropout2d(p=0.3)

    self.conv3_1 = torch.nn.Conv2d(256, 512, (3, 3), padding = 'same')
    self.conv3_2 = torch.nn.Conv2d(512, 512, (3, 3), padding = 'same')
    self.pool3 = torch.nn.AvgPool2d((8, 8), (8, 8))
    self.dropout3 = nn.Dropout2d(p=0.4)

    self.flatten = nn.Flatten()

    self.fc4 = nn.Linear(2048, 512)
    self.batchnorm_fc4 = nn.BatchNorm1d(512)
    self.dropout4 = nn.Dropout2d(p=0.4)
    self.fc5 = nn.Linear(512, num_classes)

  def forward(self, x):
    x = nn.Sequential(
      self.conv1_1,
      nn.ReLU(),
      self.conv1_2,
      nn.ReLU(),
      self.pool1,
      self.dropout1,

      self.conv2_1,
      nn.ReLU(),
      self.conv2_2,
      nn.ReLU(),
      self.pool2,
      self.dropout2,

      self.conv3_1,
      nn.ReLU(),
      self.conv3_2,
      nn.ReLU(),
      self.pool3,
      self.dropout3,
    )(x)

    x = self.flatten(x)
    x = torch.relu(self.fc4(x))
    x = self.batchnorm_fc4(x)
    x = self.dropout4(x)
    x = torch.relu(self.fc5(x))

    return x