import torch

CHANNELS = 3

class Net(torch.nn.Module):
  def __init__(self, in_channels = CHANNELS, num_classes = 5 + 1):
    super().__init__()

    self.conv1_1 = torch.nn.Conv2d(in_channels, 16, (3, 3), padding = 'same')
    self.relu1_1 = torch.nn.ReLU()
    self.conv1_2 = torch.nn.Conv2d(16, 16, (3, 3), padding = 'same')
    self.relu1_2 = torch.nn.ReLU()

    self.pool2 = torch.nn.MaxPool2d((2, 2), (2, 2))
    self.conv2_1 = torch.nn.Conv2d(16, 32, (3, 3), padding = 'same')
    self.relu2_1 = torch.nn.ReLU()
    self.conv2_2 = torch.nn.Conv2d(32, 32, (3, 3), padding = 'same')
    self.relu2_2 = torch.nn.ReLU()

    self.pool3 = torch.nn.MaxPool2d((2, 2), (2, 2))
    self.conv3_1 = torch.nn.Conv2d(32, 64, (3, 3), padding = 'same')
    self.relu3_1 = torch.nn.ReLU()
    self.conv3_2 = torch.nn.Conv2d(64, 64, (3, 3), padding = 'same')
    self.relu3_2 = torch.nn.ReLU()

    self.pool4 = torch.nn.MaxPool2d((2, 2), (2, 2))
    self.conv4_1 = torch.nn.Conv2d(64, 128, (3, 3), padding = 'same')
    self.relu4_1 = torch.nn.ReLU()
    self.conv4_2 = torch.nn.Conv2d(128, 128, (3, 3), padding = 'same')
    self.relu4_2 = torch.nn.ReLU()
    self.conv4_3 = torch.nn.Conv2d(128, 64, (3, 3), padding = 'same')
    self.relu4_3 = torch.nn.ReLU()
    self.upscale4 = torch.nn.Upsample(scale_factor = 2)

    self.conv5_1 = torch.nn.Conv2d(64, 64, (3, 3), padding = 'same')
    self.relu5_1 = torch.nn.ReLU()
    self.conv5_2 = torch.nn.Conv2d(64, 32, (3, 3), padding = 'same')
    self.relu5_2 = torch.nn.ReLU()
    self.upscale5 = torch.nn.Upsample(scale_factor = 2)

    self.conv6_1 = torch.nn.Conv2d(32, 32, (3, 3), padding = 'same')
    self.relu6_1 = torch.nn.ReLU()
    self.conv6_2 = torch.nn.Conv2d(32, 16, (3, 3), padding = 'same')
    self.relu6_2 = torch.nn.ReLU()
    self.upscale6 = torch.nn.Upsample(scale_factor = 2)

    self.conv7_1 = torch.nn.Conv2d(16, 16, (3, 3), padding = 'same')
    self.relu7_1 = torch.nn.ReLU()
    self.conv7_2 = torch.nn.Conv2d(16, 16, (3, 3), padding = 'same')
    self.relu7_2 = torch.nn.ReLU()

    self.conv8 = torch.nn.Conv2d(16, num_classes, (1, 1))  # Adjust number of output channels
    # self.softmax = torch.nn.Softmax(dim=1)  # Use Softmax across channel dimension
    # self.conv8 = torch.nn.Conv2d(16, 1, (1, 1))
    self.sigmoid8 = torch.nn.Sigmoid()

  def forward(self, x):
    block1 = torch.nn.Sequential(
        self.conv1_1,
        self.relu1_1,
        self.conv1_2,
        self.relu1_2
    )(x)
    block2 = torch.nn.Sequential(
        self.pool2,
        self.conv2_1,
        self.relu2_1,
        self.conv2_2,
        self.relu2_2,
    )(block1)
    block3 = torch.nn.Sequential(
        self.pool3,
        self.conv3_1,
        self.relu3_1,
        self.conv3_2,
        self.relu3_2,
    )(block2)
    block4 = torch.nn.Sequential(
        self.pool4,
        self.conv4_1,
        self.relu4_1,
        self.conv4_2,
        self.relu4_2,
        self.conv4_3,
        self.relu4_3,
        self.upscale4,
    )(block3) + block3
    block5 = torch.nn.Sequential(
        self.conv5_1,
        self.relu5_1,
        self.conv5_2,
        self.relu5_2,
        self.upscale5,
    )(block4) + block2
    block6 = torch.nn.Sequential(
        self.conv6_1,
        self.relu6_1,
        self.conv6_2,
        self.relu6_2,
        self.upscale6,
    )(block5) + block1
    block7 = torch.nn.Sequential(
        self.conv7_1,
        self.relu7_1,
        self.conv7_2,
        self.relu7_2,
    )(block6)
    block8 = torch.nn.Sequential(
        self.conv8,
        self.sigmoid8
    )(block7)
    return block8
