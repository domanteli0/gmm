import os
import fiftyone.utils.openimages as fouo
import fiftyone.zoo as foz
import classes as cs
from util import pickle_data
from PIL import Image
from distutils.dir_util import copy_tree

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

  if not os.path.exists(f"{output_folder}"):
    os.makedirs(f"{output_folder}")

  if not os.path.exists(f"{output_folder}/data"):
    os.makedirs(f"{output_folder}/data")

  if not os.path.exists(f"{output_folder}/labels"):
    os.makedirs(f"{output_folder}/labels")

  if not os.path.exists(f"{output_folder}/metadata"):
    os.makedirs(f"{output_folder}/metadata")

  copy_tree(f"{input_folder}/labels", f"{output_folder}/labels")
  copy_tree(f"{input_folder}/metadata", f"{output_folder}/metadata")

  for ix, filename in enumerate(os.listdir(f"{input_folder}/data")):
    print(f"Resized: {ix}")

    img_path = f"{input_folder}/data/{filename}"
    print(img_path)

    if not (img_path.endswith(".jpg") or img_path.endswith(".png")):
      print("SKIPPED")
      continue

    with Image.open(img_path) as img:
      x, y = img.size

      affordance = 0
      if x > y:
        times = MAX / (y + affordance)
      else:
        times = MAX / (x + affordance)

      img.thumbnail((int(x / times), int(y / times)), Image.LANCZOS)

      img.save(f"{output_folder}/data/{filename}")

  print(f"done with: {input_folder}")

DOWNLOAD = False
if DOWNLOAD:
  valid_ds = download("validation", max_samples=300)
  test_ds = download("test", max_samples=300)
  train_ds = download("train")

RESIZE = True
if RESIZE:
  resize_dataset('./data-lab3/test', './data-lab3-downsized/test')
  resize_dataset('./data-lab3/validation', './data-lab3-downsized/validation')
  resize_dataset('./data-lab3/train', './data-lab3-downsized/train')

PICKLE = True
if PICKLE:
  test_ds = load("test")
  valid_ds = load("validation")
  train_ds = load("train")

  pickle_data(test_ds, 'data-lab3-dyi/test.pkl', cs.classes)
  pickle_data(valid_ds, 'data-lab3-dyi/valid.pkl', cs.classes)
  pickle_data(train_ds, 'data-lab3-dyi/train.pkl', cs.classes)