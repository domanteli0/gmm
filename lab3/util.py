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

import matplotlib.pyplot as plt
import torch
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F


def plot(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            boxes = None
            masks = None
            if isinstance(img, tuple):
                img, target = img
                if isinstance(target, dict):
                    boxes = target.get("boxes")
                    masks = target.get("masks")
                elif isinstance(target, tv_tensors.BoundingBoxes):
                    boxes = target
                else:
                    raise ValueError(f"Unexpected target type: {type(target)}")
            img = F.to_image(img)
            if img.dtype.is_floating_point and img.min() < 0:
                # Poor man's re-normalization for the colors to be OK-ish. This
                # is useful for images coming out of Normalize()
                img -= img.min()
                img /= img.max()

            img = F.to_dtype(img, torch.uint8, scale=True)
            if boxes is not None:
                img = draw_bounding_boxes(img, boxes, colors="yellow", width=3)
            if masks is not None:
                img = draw_segmentation_masks(img, masks.to(torch.bool), colors=["green"] * masks.shape[0], alpha=.65)

            ax = axs[row_idx, col_idx]
            ax.imshow(img.permute(1, 2, 0).numpy(), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()

