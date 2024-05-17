from torchvision.transforms import v2
import torch

# UserWarning: The transform `ToTensor()` is deprecated and will be removed in a future release. Instead, please use `v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])`.Output is equivalent up to float precision.
to_tensor = v2.Compose([
  v2.ToImage(),
  v2.ToDtype(torch.float32, scale=True)
])

train_trans = v2.Compose([
  v2.RandomHorizontalFlip(),
  v2.RandomRotation(20),
  v2.ColorJitter(brightness = 0.4, contrast = 0.2, saturation = 0.2, hue=0.1),
  v2.RandomGrayscale(p=0.1),
  v2.RandomResizedCrop(size = (128, 128), scale = (0.75, 0.75)),
  to_tensor,
  v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

validation_trans = v2.Compose([
  v2.Resize((128, 128)),
  to_tensor,
  v2.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_trans = validation_trans

