from copy import deepcopy
from random import shuffle
from torch import Tensor
from torch.utils.data import Dataset
from pathlib import Path, PurePath
from typing import *
from PIL import Image

class MyDataSet(Dataset):
    def __init__(self, data_dir, preprocess_fn: Callable[[Tensor], Image.Image], label_map) -> None:
        self.preprocess_fn = preprocess_fn
        paths = [p for p in Path(data_dir).glob('**/images/*.jpg')]

        classes = [label_map[p.parts[-3]] for p in paths]
        paths = [p.as_posix() for p in paths]

        z = list(zip(classes, paths))
        shuffle(z)
        s = list(zip(*z))
        self.classes = s[0]
        self.paths = s[1]

    def __getitem__(self, index) -> tuple[Tensor, str]:
        with Image.open(self.paths[index]).convert('RGB') as image:
            _class = self.classes[index]
            image_processed = self.preprocess_fn(image)
            return (image_processed, _class)
    
    def __len__(self):
        return len(self.paths)
