from torch import Tensor
from torchvision.models import ResNet101_Weights
from torchvision.transforms import transforms as trans
from PIL import Image

# same as `ResNet101_Weights.DEFAULT.transform()`, except for the Normalize part
def img_trans_1(img: Image.Image) -> Tensor:
    return trans.Compose([
        trans.Resize((224, 224)),
        trans.ToTensor(),
        trans.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])(img)

def img_trans_2(img: Image.Image) -> Tensor:
    return ResNet101_Weights.transforms()