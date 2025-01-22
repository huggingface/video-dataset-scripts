import timm
import torch
import torch.nn as nn
import torchvision.transforms as T

MODEL, TRANSFORMS = None, None


def load_watermark_laion(device, model_path):
    global MODEL, TRANSFORMS
    TRANSFORMS = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    MODEL = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
    MODEL.classifier = nn.Sequential(
        nn.Linear(in_features=1536, out_features=625),
        nn.ReLU(),
        nn.Dropout(p=0.3),
        nn.Linear(in_features=625, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    state_dict = torch.load(model_path, weights_only=True)
    MODEL.load_state_dict(state_dict)
    MODEL.eval().to(device)


@torch.no_grad
def run_watermark_laion(image):
    if not isinstance(image, list):
        image = [image]
    pixel_values = torch.stack([TRANSFORMS(_image) for _image in image])
    return nn.functional.softmax(MODEL(pixel_values), dim=1)[:, 0]
