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

    # (2000x512 and 32768x512)
    # self.fc4 = nn.Linear(512 * (DIM // 8) * (DIM // 8), 512)

    # (364x2048 and 512x256)
    # (2000x512 and 2048x364)
    # self.fc4 = nn.Linear(2048, 364)
    # (2000x512 and 2048x500)
    self.fc4 = nn.Linear(2048, 512)
    self.batchnorm_fc4 = nn.BatchNorm1d(512)
    self.dropout4 = nn.Dropout2d(p=0.4)

    self.fc5 = nn.Linear(512, num_classes)

    # self.fc4_3 = nn.Linear(2048, num_classes)

    # self.conv1 = nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=64, kernel_size=3, padding=1)
    # self.batchnorm1 = nn.BatchNorm2d(64)
    # self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
    # self.batchnorm2 = nn.BatchNorm2d(64)
    # self.maxpool1 = nn.MaxPool2d(kernel_size=2)
    # self.dropout1 = nn.Dropout2d(p=0.2)
# 
    # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
    # self.batchnorm3 = nn.BatchNorm2d(128)
    # self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
    # self.batchnorm4 = nn.BatchNorm2d(128)
    # self.maxpool2 = nn.MaxPool2d(kernel_size=2)
    # self.dropout2 = nn.Dropout2d(p=0.3)
# 
    # self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
    # self.batchnorm5 = nn.BatchNorm2d(256)
    # self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
    # self.batchnorm6 = nn.BatchNorm2d(256)
    # self.maxpool3 = nn.MaxPool2d(kernel_size=2)
    # self.dropout3 = nn.Dropout2d(p=0.4)
# 
    # self.flatten = nn.Flatten()
    # self.fc1 = nn.Linear(9216, 2048)
    # self.fc2 = nn.Linear(2048, 256)
    # self.batchnorm_fc = nn.BatchNorm1d(256)
    # self.dropout_fc = nn.Dropout(p=0.5)
    # self.fc3 = nn.Linear(256, num_classes)

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

    # # self.fc4 = nn.Linear(2048, 512)
    # # self.batchnorm_fc4 = nn.BatchNorm1d(512)
    # # self.dropout4 = nn.Dropout2d(p=0.4)
    # # self.fc5 = nn.Linear(512, num_classes)

    # # x = self.flatten(x)
    # # x = torch.relu(self.fc4(x))
    # # x = self.batchnorm_fc4(x)
    # # x = self.dropout4(x)
    # # x = torch.relu(self.fc5(x))

    x = self.flatten(x)
    # x = nn.ReLU(self.fc4(x))
    # x = self.batchnorm_fc4(x)
    x = torch.relu(self.fc4(x))

    return x
    # x = torch.relu(self.conv1(x))
    # x = self.batchnorm1(x)
    # x = torch.relu(self.conv2(x))
    # x = self.batchnorm2(x)
    # x = self.maxpool1(x)
    # x = self.dropout1(x)
# 
    # x = torch.relu(self.conv3(x))
    # x = self.batchnorm3(x)
    # x = torch.relu(self.conv4(x))
    # x = self.batchnorm4(x)
    # x = self.maxpool2(x)
    # x = self.dropout2(x)
# 
    # x = torch.relu(self.conv5(x))
    # x = self.batchnorm5(x)
    # x = torch.relu(self.conv6(x))
    # x = self.batchnorm6(x)
    # x = self.maxpool3(x)
    # x = self.dropout3(x)
# 
    # x = self.flatten(x)
    # x = torch.relu(self.fc1(x))
    # x = torch.relu(self.fc2(x))
    # x = self.batchnorm_fc(x)
    # x = self.dropout_fc(x)
    # x = self.fc3(x)
# 
    # return x