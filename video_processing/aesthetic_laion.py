import torch
import torch.nn as nn
from transformers import CLIPVisionModelWithProjection, CLIPProcessor

MODEL = None


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(768, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    @torch.no_grad()
    def forward(self, embed):
        return self.layers(embed)


class AestheticScorer(torch.nn.Module):
    def __init__(self, dtype, path):
        super().__init__()
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.mlp = MLP()
        state_dict = torch.load(path, weights_only=True, map_location=torch.device('cpu'))
        self.mlp.load_state_dict(state_dict)
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images):
        device = next(self.parameters()).device
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.dtype).to(device) for k, v in inputs.items()}
        embed = self.clip(**inputs)[0]
        # normalize embedding
        embed = embed / torch.linalg.vector_norm(embed, dim=-1, keepdim=True)
        return self.mlp(embed).squeeze(1)


def load_aesthetic_laion(model_path, device, dtype):
    global MODEL
    dtype = getattr(torch, dtype)
    MODEL = AestheticScorer(dtype=dtype, path=model_path).to(device)


@torch.no_grad
def run_aesthetic_laion(image):
    if not isinstance(image, list):
        image = [image]
    return MODEL(image)
