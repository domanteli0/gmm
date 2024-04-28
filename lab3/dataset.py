from typing import Iterable, cast
from fiftyone.core.labels import Detections
from fiftyone.core.labels import Detection
import fiftyone.core.dataset as fod
import fiftyone.core.sample as fos
import fiftyone.utils.openimages as fouo
import torch
import torchvision.transforms as trans
import numpy as np
from PIL import Image
import cv2

import lab3.classes as cs

class FiftyOneDataset(torch.utils.data.Dataset):
  def __init__(self, importer: fouo.OpenImagesV6DatasetImporter, transforms):
    self.transform = transforms
    self.img_paths = [filepath for filepath, _, s in importer if s is not None]
    self.masks = [aggregate_detections(segs) for _, _, segs in importer if segs is not None]

  def __len__(self):
    return len(self.img_paths)

  def __getitem__(self, idx):
    img  = self.img_paths[idx]
    mask = self.masks[idx]

    img  = Image.open(img).convert('RGB')
    mask = trans.functional.to_tensor(mask)

    # TODO:
    # img, mask = self.transform(img), self.transform(mask)
    img, mask = self.transform(img), mask

    return img, mask

def resize_mask(mask, bbox, target_size = (128, 128)):
  x_start = int(bbox[0] * target_size[1])
  y_start = int(bbox[1] * target_size[0])
  x_end = int((bbox[0] + bbox[2]) * target_size[1])
  y_end = int((bbox[1] + bbox[3]) * target_size[0])

  return cv2.resize(mask, (x_end - x_start, y_end - y_start), interpolation = cv2.INTER_NEAREST), x_start, y_start

def aggregate_detections(segmentations: Detections, target_size: tuple[int, int] = (128, 128)):
  aggr_mask = np.zeros((*target_size, cs.num_classes), dtype = np.uint8)
  aggr_mask[..., cs.classes_dict["Background"]] = 1

  seg: Detection
  for seg in segmentations.detections:
    # print(f"SEG: {seg}")

    if seg.label not in cs.classes_dict.keys():
      continue

    label = seg.label
    mask = seg.mask.astype(np.uint8)
    bbox = seg.bounding_box

    resized_mask, x_start, y_start = resize_mask(mask, bbox)

    for k in range(resized_mask.shape[0]):
      for j in range(resized_mask.shape[1]):
        if resized_mask[k, j]:
          aggr_mask[y_start + k, x_start + j, :] = 0
          aggr_mask[y_start + k, x_start + j, cs.classes_dict[label]] = 1

  return aggr_mask