from pathlib import Path
from openimages.download import download_dataset
import os

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

classes = ["Container", "Skyscraper", "Wheelchair", "Helmet"]

def download_if_not_found():
    for (class_dir, _class) in [(Path(f"{data_dir}/{_class.lower()}"), _class) for _class in classes]:
        if not class_dir.exists():
            print(f"Downloading: {_class}")
            download_dataset(
                data_dir,
                [_class],
                limit = 340
            )