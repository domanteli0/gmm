import torch
from torch import tensor as t

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

# Normalizes masks PER singular sample
def normalize_masks_in_a_batch(mask_batch: torch.Tensor) -> torch.Tensor:
  BATCH_SIZE = t(mask_batch.shape)[0]
  CLASS_NO = t(mask_batch.shape)[1]
  DIM = t(mask_batch.shape)[2]

  vals, _ = mask_batch.max(dim = -1)[0].max(dim = -1)[0].max(dim = -1)
  vals = vals ** -1
  t2 = vals.repeat_interleave(CLASS_NO * DIM * DIM).view((BATCH_SIZE, CLASS_NO, DIM, DIM))
  return t2 * mask_batch

# Normalizes masks on a class by class basis
def normalize_masks(masks: torch.Tensor) -> torch.Tensor:
  CLASS_NO = t(masks.shape)[0]
  DIM = t(masks.shape)[1]

  vals, _ = masks.max(dim = -1)[0].max(dim = -1)
  vals = vals ** -1
  t2 = vals.repeat_interleave(DIM * DIM).view((CLASS_NO, DIM, DIM))
  return t2 * masks

