from typing import Iterable, cast
import torch
import torchvision.transforms.v2 as trans
from PIL import Image
import cv2
import torch as np
import matplotlib.pyplot as plt
import torchvision.datasets as tvd

import lab3.util as lu
import lab3.classes as cs
import lab3.trans as lt
from torchvision.tv_tensors import Mask as TVMask, Image as TVImage
from lab3.net import DIM

# class FiftyOneDataset(torch.utils.data.Dataset):
class FiftyOneDataset(tvd.VisionDataset):
  def __init__(self, filepath, transforms, split = 'train'):
    # super(FiftyOneDataset).__init__()

    data = lu.unpickle_data(filepath)

    self.transform = transforms
    # self.img_paths = [s['image'] for s in set]
    # self.masks = [s['segmentations'] for s in set]
    self.data = data
    self.split = split

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    image = Image.open(f"data-lab3-downsized/{self.split}/data/{self.data[idx]['image']}").convert('RGB')
    img_x, img_y = image.size
    masks = self.data[idx]['segmentations']
    masks_as_tensor = mk_zero_mask(img_x, img_y)
    for pair in masks:
      label = pair['label']
      mask = pair['mask']
      masks_as_tensor[cs.classes_dict[label]] = torch.tensor(mask)

    image = TVImage(image)
    masks = TVMask(masks_as_tensor)

    image, masks = self.transform(image, masks)

    return image, masks

def mk_zero_mask(img_x, img_y):
  return torch.zeros((cs.num_classes, img_y, img_x))

def resize_mask(mask, bbox, target_size=(DIM, DIM)):
  x_start = int(bbox[0] * target_size[1])
  y_start = int(bbox[1] * target_size[0])
  x_end = int((bbox[0] + bbox[2]) * target_size[1])
  y_end = int((bbox[1] + bbox[3]) * target_size[0])

  return cv2.resize(mask, (x_end - x_start, y_end - y_start), interpolation=cv2.INTER_LINEAR), x_start, y_start

#
# def aggregate_detections(segmentations: Detections, target_size: tuple[int, int] = (128, 128)):
#   aggr_mask = np.zeros((*target_size, cs.num_classes), dtype=np.uint8)
#   aggr_mask[..., cs.classes_dict["Background"]] = 1
#
#   seg: Detection
#   for seg in segmentations.detections:
#     print(f"SEG: {seg}")
#
#     if seg.label not in cs.classes_dict.keys():
#       continue
#
#     label = seg.label
#     mask = seg.mask.astype(np.uint8)
#     bbox = seg.bounding_box
#
#     resized_mask, x_start, y_start = resize_mask(mask, bbox)
#
#     for k in range(resized_mask.shape[0]):
#       for j in range(resized_mask.shape[1]):
#         if resized_mask[k, j]:
#           aggr_mask[y_start + k, x_start + j, :] = 0
#           aggr_mask[y_start + k, x_start + j, cs.classes_dict[label]] = 1
#
#   return aggr_mask
