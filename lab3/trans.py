import torchvision.transforms.v2 as trans

train_trans = trans.Compose([
  trans.Resize((128, 128)),
  trans.ToTensor(),
  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

validation_trans = trans.Compose([
  trans.Resize((128, 128)),
  trans.ToTensor(),
  trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

test_trans = validation_trans
