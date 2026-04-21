import torch
import torch.nn as nn
from torchvision import transforms
import io
from PIL import Image
from fastapi import HTTPException, APIRouter, File, UploadFile

class VisionLogic(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 16 * 16, 128),
        nn.ReLU(),
        nn.Linear(128, 6)
    )

  def forward(self, x):
    x = self.first(x)
    return self.second(x)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VisionLogic().to(device)
model.load_state_dict(torch.load('models/trash_model.pth', map_location=device))
model.eval()

labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

trash_router = APIRouter(prefix='/trash', tags=['CV'])

@trash_router.post('/')
async def predict_img(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        if not img_data:
            raise HTTPException(detail='Upload the file!', status_code=400)
        img = Image.open(io.BytesIO(img_data)).convert('RGB')
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(tensor_img)
            pred = predict.argmax(dim=1).item()

            return {'Prediction': labels[pred]}
    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)