import matplotlib.pyplot as plt
import numpy as np

def display_img_with_masks(ax, img, masks):
  # TODO:
  cmaps = ['']

  img = img.permute(1, 2, 0).numpy()

  ax.imshow(img)
  for ix in range(masks[:,0,0].shape[0]):
    mask = masks[ix,:,:].numpy()
  
    if np.max(mask) == 0:
      mask = np.zeros((128, 128))
    else:
      mask = mask / np.max(mask)

    mask = np.ma.masked_where(mask == 0, mask)
    ax.imshow(mask,alpha=0.7)

