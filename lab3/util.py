import torch
import torch as np
import pickle as p
import itertools

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

    # Sorting the data by label
    seg.detections.sort(key=lambda x: x.label)
    for label, segs in itertools.groupby(seg.detections, key=lambda d: d.label):
      if label not in classes:
        continue

      print(list(segs))
      print(label)

      masks  = [s['mask'] for s in segs]
      bboxes = [s['bounding_box'] for s in segs]

      mask, bbox = big_mask_from_many_smol_masks(masks, bboxes)

      if mask.max() != 0:
        seg = {
          "label": label,
          "bbox": bbox,
          "mask": mask
        }
        segmentations['segmentations'].append(seg)

    set.append(segmentations)

  with open(filepath, 'wb') as file:
    p.dump(set, file)

  print(f"done: {ix}")

def unpickle_data(filepath):
  with open(filepath, 'wb') as file:
    return p.load(set, file)

def bix_bbox_from_smol_bboxes(bboxes):
  """
  arg and return bbox format:

  (x1, y1, x2, y2), where

  x1, y1 - top-right corner

  x2 - distance from x1 to the right

  y2 - distance from y1 to the bottom
  """
  min_x, min_y = 0, 0
  max_x, max_y = 0, 0

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

def big_mask_from_many_smol_masks(masks, bboxes):
  """
  arg and return bbox format:

  (x1, y1, x2, y2), where

  x1, y1 - top-right corner

  x2 - distance from x1 to the right

  y2 - distance from y1 to the bottom
  """
  new_bbox = bix_bbox_from_smol_bboxes(bboxes)
  min_x, min_y, max_x, max_y = new_bbox
  mask = np.zeros((min_x + max_x, min_y +  max_y))
  # mask = np.zeros((max_y + 1, max_y + 1))

  for small_mask, bbox in zip(masks, bboxes):
    x1, y1, x2, y2 = bbox
    x2, y2 = x2+x1, y2+y1
    mask[y1:y2, x1:x2] += small_mask

  return mask, new_bbox

def pad_mask(mask, orig_bbox, new_dim):
  zeros = np.zeros(new_dim[0], new_dim[1])
  new_bbox = (0, 0, new_dim[0], new_dim[1])

  return big_mask_from_many_smol_masks(
    masks  = [zeros, mask],
    bboxes = [new_bbox, orig_bbox],
  )
