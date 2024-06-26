from torchvision.transforms import transforms as trans

train_trans = trans.Compose([
  trans.RandomHorizontalFlip(),
  trans.RandomRotation(20),
  trans.ColorJitter(brightness = 0.4, contrast = 0.2, saturation = 0.2, hue=0.1),
  trans.RandomGrayscale(p=0.1),
  trans.RandomResizedCrop(size = (64, 64), scale = (0.75, 0.75)),
  trans.ToTensor(),
  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

validation_trans = trans.Compose([
  trans.Resize((64, 64)),
  trans.ToTensor(),
  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_trans = validation_trans