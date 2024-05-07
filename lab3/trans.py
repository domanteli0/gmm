import torchvision.transforms.v2 as trans
import torch

# UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.Output is equivalent up to float precision.
to_tensor = trans.Compose([
  trans.ToImage(),
  trans.ToDtype(torch.float32, scale=True)
])

train_trans = trans.Compose([
  trans.RandomHorizontalFlip(),
  trans.RandomRotation(20),
  trans.ColorJitter(brightness = 0.4, contrast = 0.2, saturation = 0.2, hue=0.1),
  trans.RandomGrayscale(p=0.1),
  trans.RandomResizedCrop(size = (128, 128), scale = (0.75, 0.75)),
  to_tensor,
  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

validation_trans = trans.Compose([
  trans.Resize((128, 128)),
  to_tensor,
  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_trans = validation_trans

