import os
import fiftyone.utils.openimages as fouo
import fiftyone.zoo as foz
import classes as cs
from util import pickle_data
from PIL import Image
import shutil

def download(split="train", max_samples: int = 2000):
  return foz.load_zoo_dataset(
    "open-images-v6",
    split=split,
    label_types=["segmentations", "detections"],
    classes=cs.classes_no_background,
    max_samples=max_samples,
    dataset_dir="data-lab3",
    dataset_name=f"open-images-v6-{split}"
  )


def load(split="train"):
  dataset = fouo.OpenImagesV6DatasetImporter(
    dataset_dir=f"data-lab3-downsized/{split}",
    label_types="segmentations"
  )

  dataset.setup()

  return dataset


def resize_dataset(input_folder, output_folder):
  os.makedirs(output_folder, exist_ok=True)
  MAX = 256

  for ix, filename in enumerate(os.listdir(input_folder)):
    print(f"Resized: {ix}")
    # Create the full file path
    file_path = os.path.join(input_folder, filename)

    # Open an image file
    with Image.open(file_path) as img:
      x, y = img.size

      affordance = 0
      if x > y:
        times = MAX / (y + affordance)
      else:
        times = MAX / (x + affordance)

      img.thumbnail((int(x / times), int(y / times)), Image.LANCZOS)

      # Save it to the output folder
      output_path = os.path.join(output_folder, filename)
      img.save(output_path)


DOWNLOAD = False
if DOWNLOAD:
  train_ds = download("train")
  valid_ds = download("validation", max_samples=300)
  test_ds = download("test", max_samples=300)

RESIZE = True
if RESIZE:
  resize_dataset('./data-lab3/test/data/', './data-lab3-downsized/test/data/')
  resize_dataset('./data-lab3/validation/data/', './data-lab3-downsized/validation/data/')
  resize_dataset('./data-lab3/train/data/', './data-lab3-downsized/train/data/')

PICKLE = True
if PICKLE:
  train_ds = load("train")
  test_ds = load("test")
  valid_ds = load("validation")

  pickle_data(train_ds, 'data-lab3-dyi/train.pkl', cs.classes)
  pickle_data(test_ds, 'data-lab3-dyi/test.pkl', cs.classes)
  pickle_data(valid_ds, 'data-lab3-dyi/valid.pkl', cs.classes)
