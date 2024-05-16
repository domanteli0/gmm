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

def display_masks(ax, masks):
  for ix in range(masks[:,0,0].shape[0]):
    mask = masks[ix,:,:].numpy()

    if np.max(mask) == 0:
      mask = np.zeros((128, 128))
    else:
      mask = mask / np.max(mask)

    mask = np.ma.masked_where(mask == 0, mask)
    ax.imshow(mask,alpha=0.7)


# import matplotlib.pyplot as plt
# import numpy as np

# def show_image(img, mask):
#   plt.clf()
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (4 * 2, 3), sharex=True, sharey=True)

#   ax1.imshow(img.permute(((2, 1, 0))) * 0.5 + 0.5)
#   # ax2.imshow(mask.permute((2, 1, 0)), vmin = 0, vmax = 1)
#   # ax1.imshow(img * 0.5 + 0.5)
#   ax2.imshow(mask, vmin = 0, vmax = 1)

#   plt.show()

def mask_to_rgb_image(mask, classes):
  # black, white, green, red, blue, cyan
  colors = [(0, 0, 0), (255, 230, 255), (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
  # magenta, yellow, orange, purple
  colors.extend([(255, 0, 255), (255, 255, 0),  (255, 165, 0), (128, 0, 128)])
  colors = colors[:len(classes)]

  height, width, n_classes = mask.shape
  rgb_images = []

  rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
  for c in range(n_classes):
    # For each class, apply the color to the mask where that class is present
    rgb_image[mask[:, :, c] == 1] = colors[c]
  
  return rgb_image

def display_img_with_masks_nouveau(img, masks, classes):
  fig, axes = plt.subplots(1, 6, figsize=(15, 15))
  for i in range(len(classes)):
    axes[i].imshow(masks[i], cmap='gray')
    axes[i].set_title(classes[i])
    axes[i].axis('off')
  plt.show()
  plt.imshow(img.permute(1, 2, 0).numpy())

