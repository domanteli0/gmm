import torch
from torch import nn

CHANNELS = 3

def def_conv(in_channels, out_channels):
  return nn.Sequential(
    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True)
  )

class Net(torch.nn.Module):
  def __init__(self, in_channels = CHANNELS, num_classes = 5 + 1):
    super().__init__()

    self.conv1 = def_conv(in_channels, 32)
    self.pool1 = nn.MaxPool2d(2)
    self.drop1 = nn.Dropout(0.2)

    self.conv2 = def_conv(32, 64)
    self.pool2 = nn.MaxPool2d(2)
    self.drop2 = nn.Dropout(0.3)

    self.conv3 = def_conv(64, 128)
    self.pool3 = nn.MaxPool2d(2)
    self.drop3 = nn.Dropout(0.4)

    self.conv4 = def_conv(128, 256)
    self.drop4 = nn.Dropout(0.5)

    self.up5 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.conv5 = def_conv(256, 128)

    self.up6 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv6 = def_conv(128, 64)

    self.up7 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.conv7 = def_conv(64, 32)

    # Output Layer
    self.conv8 = torch.nn.Conv2d(32, num_classes, (1, 1))  # Adjust number of output channels
    self.softmax = torch.nn.Softmax(dim = 0)  # Use Softmax across channel dimension
    # self.conv8 = torch.nn.Conv2d(16, 1, (1, 1))
    # self.sigmoid8 = torch.nn.Sigmoid()

  def forward(self, x):
    # Contracting Path
    c1 = self.conv1(x)
    p1 = self.pool1(c1)
    d1 = self.drop1(p1)

    c2 = self.conv2(d1)
    p2 = self.pool2(c2)
    d2 = self.drop2(p2)

    c3 = self.conv3(d2)
    p3 = self.pool3(c3)
    d3 = self.drop3(p3)

    # Bottom layer
    c4 = self.conv4(d3)
    d4 = self.drop4(c4)

    # Expanding Path
    u5 = self.up5(d4)
    merge5 = torch.cat([u5, c3], dim=1)
    c5 = self.conv5(merge5)

    u6 = self.up6(c5)
    merge6 = torch.cat([u6, c2], dim=1)
    c6 = self.conv6(merge6)

    u7 = self.up7(c6)
    merge7 = torch.cat([u7, c1], dim=1)
    c7 = self.conv7(merge7)

    c8 = self.conv8(c7)
    out = self.softmax(c8)
    return out
