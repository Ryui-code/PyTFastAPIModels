from fastapi import UploadFile, APIRouter, File, HTTPException
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

labels = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

class ConvNeXt(nn.Module):
    def __init__(self):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.model = models.convnext_tiny(weights=weights)
        self.model.classifier[2] = nn.Linear(768, len(labels))

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNeXt().to(device)
model.load_state_dict(torch.load("models/image_scene_model.pth", map_location=device))
model.eval()

scene_router = APIRouter(prefix="/scene", tags=['Image to Text'])

@scene_router.post("/")
async def predict_img(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        if not image_data:
            raise HTTPException(status_code=400, detail="Upload the file!")
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(tensor_img)
            pred = predict.argmax(dim=1).item()

        return {"Prediction": labels[pred]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))