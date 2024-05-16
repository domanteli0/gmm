from typing import List

import torch
import numpy as np
import pickle as p
import itertools
from PIL import Image
import cv2


def seconds_to_time(seconds):
  s = int(seconds) % 60
  m = int(seconds) // 60
  if m < 1:
    return f'{s}s'
  h = m // 60
  m = m % 60
  if h < 1:
    return f'{m}m{s}s'
  return f'{h}h{m}m{s}s'


# # Normalizes masks PER singular sample
# def normalize_masks_in_a_batch(mask_batch: torch.Tensor) -> torch.Tensor:
#   BATCH_SIZE = t(mask_batch.shape)[0]
#   CLASS_NO = t(mask_batch.shape)[1]
#   DIM = t(mask_batch.shape)[2]
#
#   vals, _ = mask_batch.max(dim = -1)[0].max(dim = -1)[0].max(dim = -1)
#   vals = vals ** -1
#   t2 = vals.repeat_interleave(CLASS_NO * DIM * DIM).view((BATCH_SIZE, CLASS_NO, DIM, DIM))
#   return t2 * mask_batch
#
# # Normalizes masks on a class by class basis
# def normalize_masks(masks: torch.Tensor) -> torch.Tensor:
#   CLASS_NO = t(masks.shape)[0]
#   DIM = t(masks.shape)[1]
#
#   vals, _ = masks.max(dim = -1)[0].max(dim = -1)
#   vals = vals ** -1
#   t2 = vals.repeat_interleave(DIM * DIM).view((CLASS_NO, DIM, DIM))
#   return t2 * masks

def pickle_data(dataset, filepath, classes):
  ix = 1
  set = []
  for i, q, seg in dataset:
    if not hasattr(seg, 'detections'):
      continue
    ix += 1

    img = i.split('/')[-1]
    segmentations = {
      "image": img,
      "segmentations": []
    }

    img = Image.open(i).convert('RGB')
    img_x, img_y = img.size

    # Sorting the data by label
    seg.detections.sort(key=lambda x: x.label)
    for label, segs in itertools.groupby(seg.detections, key=lambda d: d.label):
      segs = list(segs)
      if label not in classes:
        continue

      masks = [s.mask.astype(int) for s in segs]
      bboxes = [to_target_shape(s.bounding_box, img.size) for s in segs]

      # https://stackoverflow.com/a/64617349
      # mask = np.zeros((height, width))
      # TODO: rename
      zero_mask = np.zeros((img_y, img_x), dtype=int)
      for mask, bbox in zip(masks, bboxes):
        mask = resize_mask(mask, bbox)
        x1, y1, x2, y2 = bbox
        x2 += x1
        y2 += y1

        zero_mask[y1:y2, x1:x2] = mask

      if zero_mask.max() != 0:
        seg = {
          "label": label,
          "mask": zero_mask
        }
        segmentations['segmentations'].append(seg)

    set.append(segmentations)

  with open(filepath, 'wb') as file:
    p.dump(set, file)

  print(f"done: {ix}")


def resize_mask(mask, bounding_box):
  _, _, x2, y2 = bounding_box
  return cv2.resize(mask,
                    dsize=(x2, y2),
                    interpolation=cv2.INTER_NEAREST)


def to_target_shape(bbox, target_shape):
  bbox[0] *= target_shape[0]
  bbox[1] *= target_shape[1]
  bbox[2] *= target_shape[0]
  bbox[3] *= target_shape[1]

  return int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])


def unpickle_data(filepath):
  with open(filepath, 'rb') as file:
    return p.load(file)


def bix_bbox_from_smol_bboxes(bboxes):
  """
  arg and return bbox format:

  (x1, y1, x2, y2), where

  x1, y1 - top-right corner

  x2 - distance from x1 to the right

  y2 - distance from y1 to the bottom
  """
  min_x, min_y = 1., 1.
  max_x, max_y = 0., 0.

  for bbox in bboxes:
    x1, y1, x2, y2 = bbox
    x2 = x1 + x2
    y2 = y1 + y2

    min_x = min(min_x, x1)
    min_y = min(min_y, y1)
    max_x = max(max_x, x2)
    max_y = max(max_y, y2)

  return min_x, min_y, max_x - min_x, max_y - min_y


# masks = [ np.array([[1,1], [1,1]]), np.array([[0,0], [0,1]]) ]
# bboxes = [(1,1,2,2), (2,2,2,2)]

def big_mask_from_many_smol_masks(masks, start_pos):
  """
  arg and return bbox format:

  (x1, y1, x2, y2), where

  x1, y1 - top-right corner

  x2 - distance from x1 to the right

  y2 - distance from y1 to the bottom
  """
  new_bbox = bix_bbox_from_smol_bboxes(bboxes)
  mask = zero_mask_from_smol_masks(masks)
  # mask = np.zeros((max_y + 1, max_y + 1))
  print(f"MASK SHAPE: {mask.shape}")

  for small_mask, bbox in zip(masks, bboxes):
    x1, y1, _, _ = bbox
    x1 = int(x1)
    y1 = int(y1)
    x2, y2 = x1 + small_mask.shape[1], y1 + small_mask.shape[0]
    print(f"small_mask shape {small_mask.shape}")
    print(f"(,,,,) {(x1, y1, x2, y2)}")
    mask[y1:y2, x1:x2] += small_mask

  return mask, new_bbox


def zero_mask_from_smol_masks(masks):
  x, y = 0, 0
  for m in masks:
    x = max(m.shape[0], x)
    y = max(m.shape[1], y)

  return np.zeros((x, y))


def pad_mask(mask_, bbox, img_size):
  padded = np.zeros((img_size[1], img_size[0]))

  x1, y1, x2, y2 = bbox

  padded[y1:y2, x1:x2] += mask_
  padded = ((padded >= 1) * 255.0).astype(int)


def unzipListIntoList(xs: List):
  return list(zip(*xs))
