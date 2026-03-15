import torch
import torch.nn as nn
from torchvision import transforms
import io
from PIL import Image
from fastapi import HTTPException, APIRouter, File, UploadFile

class Vgg16Logic(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 1 * 1, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, layer):
        layer = self.first(layer)
        return self.second(layer)

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Vgg16Logic()
model.load_state_dict(torch.load('models/vgg_model.pth', map_location=device))
model.to(device)
model.eval()

labels = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

cifar_router = APIRouter(prefix='/cifar')

@cifar_router.post('/')
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