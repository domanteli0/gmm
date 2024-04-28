import matplotlib.pyplot as plt
import numpy as np

def show_image(img, mask):
  plt.clf()
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize= (4 * 2, 3), sharex=True, sharey=True)

  ax1.imshow(img.permute(((2, 1, 0))) * 0.5 + 0.5)
  # ax2.imshow(mask.permute((2, 1, 0)), vmin = 0, vmax = 1)
  # ax1.imshow(img * 0.5 + 0.5)
  ax2.imshow(mask, vmin = 0, vmax = 1)

  plt.show()

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