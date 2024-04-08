import torch.nn as nn
import torch

DIM = 32
NUM_CHANNELS = 3
NUM_CLASSES = 4

class Model(nn.Module):
	def __init__(self, num_classes = NUM_CLASSES):
		super(Model, self).__init__()
		self.softmax = nn.Softmax()

		self.conv1 = nn.Conv2d(in_channels=NUM_CHANNELS, out_channels=32, kernel_size=3, padding=1)
		self.batchnorm1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1)
		self.batchnorm2 = nn.BatchNorm2d(32)
		self.maxpool1 = nn.MaxPool2d(kernel_size=2)
		self.dropout1 = nn.Dropout2d(p=0.2)

		self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
		self.batchnorm3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
		self.batchnorm4 = nn.BatchNorm2d(64)
		self.maxpool2 = nn.MaxPool2d(kernel_size=2)
		self.dropout2 = nn.Dropout2d(p=0.3)

		self.conv5 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
		self.batchnorm5 = nn.BatchNorm2d(128)
		self.conv6 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
		self.batchnorm6 = nn.BatchNorm2d(128)
		self.maxpool3 = nn.MaxPool2d(kernel_size=2)
		self.dropout3 = nn.Dropout2d(p=0.4)

		self.conv7 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
		self.batchnorm7 = nn.BatchNorm2d(256)
		self.conv8 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
		self.batchnorm8 = nn.BatchNorm2d(256)
		self.maxpool4 = nn.MaxPool2d(kernel_size=2)
		self.dropout4 = nn.Dropout2d(p=0.45)

		self.flatten = nn.Flatten()
		self.fc1 = nn.Linear(1024, 256)
		# (2000x1024 and 4096x128)
		# self.fc1 = nn.Linear(256 * (DIM // 8) * (DIM // 8), 256)
		self.batchnorm_fc = nn.BatchNorm1d(256)
		self.dropout_fc = nn.Dropout(p=0.5)
		self.fc2 = nn.Linear(256, num_classes)

	def forward(self, x):
		x = torch.relu(self.conv1(x))
		x = self.batchnorm1(x)
		x = torch.relu(self.conv2(x))
		x = self.batchnorm2(x)
		x = self.maxpool1(x)
		x = self.dropout1(x)

		x = torch.relu(self.conv3(x))
		x = self.batchnorm3(x)
		x = torch.relu(self.conv4(x))
		x = self.batchnorm4(x)
		x = self.maxpool2(x)
		x = self.dropout2(x)

		x = torch.relu(self.conv5(x))
		x = self.batchnorm5(x)
		x = torch.relu(self.conv6(x))
		x = self.batchnorm6(x)
		x = self.maxpool3(x)
		x = self.dropout3(x)

		x = torch.relu(self.conv7(x))
		x = self.batchnorm7(x)
		x = torch.relu(self.conv8(x))
		x = self.batchnorm8(x)
		x = self.maxpool4(x)
		x = self.dropout4(x)

		x = self.flatten(x)
		x = torch.relu(self.fc1(x))
		x = self.batchnorm_fc(x)
		x = self.dropout_fc(x)
		x = self.fc2(x)

		return x