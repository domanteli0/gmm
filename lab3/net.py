import torch
from torch import nn
from torch import cat as c

CHANNELS = 3
DIM = 224

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
    self.drop1 = nn.Dropout(0.1)

    self.conv2 = def_conv(32, 64)
    self.pool2 = nn.MaxPool2d(2)
    self.drop2 = nn.Dropout(0.2)

    self.conv3 = def_conv(64, 128)
    self.pool3 = nn.MaxPool2d(2)
    self.drop3 = nn.Dropout(0.3)

    self.conv4 = def_conv(128, 256)
    self.pool4 = nn.MaxPool2d(2)
    self.drop4 = nn.Dropout(0.4)

    # Bottleneck
    self.conv5 = def_conv(256, 512)
    self.drop5 = nn.Dropout(0.5)

    # Going down from here in on
    self.up6 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
    self.conv6 = def_conv(512, 256)

    self.up7 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.conv7 = def_conv(256, 128)

    self.up8 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv8 = def_conv(128, 64)

    self.up9 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.conv9 = def_conv(64, 32)

    # Output Layer
    self.conv10 = torch.nn.Conv2d(32, num_classes, (1, 1))
    self.softmax = torch.nn.Softmax(dim = 1)  # Use Softmax across channel dimension

    # self.conv8 = torch.nn.Conv2d(16, 1, (1, 1))
    # self.sigmoid8 = torch.nn.Sigmoid()

  def forward(self, x):
    # Contracting Path
    c1 = self.conv1(x)
    p1 = self.pool1(c1)
    d1 = self.drop1(p1)

    c2 = self.conv2(d1)
    p2 = self.pool2(c2)
    # d2 = self.drop2(p2)

    c3 = self.conv3(p2)
    p3 = self.pool3(c3)
    # d3 = self.drop3(p3)

    c4 = self.conv4(p3)
    p4 = self.pool4(c4)
    # d4 = self.drop4(p4)

    # Bottom layer
    c5 = self.conv5(p4)
    # d5 = self.drop5(c5)

    # Expanding Path
    u6 = self.up6(c5)
    # print(f"u6: {u6.shape}  | c4: {c4.shape}")
    c6 = self.conv6(c([u6, c4], dim = 1))

    u7 = self.up7(c6)
    # print(f"u7: {u7.shape}  | c3: {c3.shape}")
    c7 = self.conv7(c([u7, c3], dim = 1))

    u8 = self.up8(c7)
    # print(f"u8: {u8.shape}  | c2: {c2.shape}")
    c8 = self.conv8(c([u8, c2], dim = 1))

    u9 = self.up9(c8)
    # print(f"u9: {u9.shape}  | c1: {c1.shape}")
    c9 = self.conv9(c([u9, c1], dim = 1))

    c10 = self.conv10(c9)
    out = self.softmax(c10)
    return out
