import torch
from torchvision import transforms
import io
from PIL import Image
from fastapi import HTTPException, File, UploadFile, APIRouter
import torch.nn as nn

class NNLogic(nn.Module):
    def __init__(self):
        super().__init__()

        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.first(x)
        return self.second(x)

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NNLogic()
model.load_state_dict(torch.load('models/model (1).pth', map_location=device))
model.to(device)
model.eval()

labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

fashion_router = APIRouter(prefix='/fashion', tags=['Image to Text'])

@fashion_router.post('/')
async def predict_img(file: UploadFile = File(...)):
    try:
        img_data = await file.read()
        if not img_data:
            raise HTTPException(detail='Upload the file!', status_code=400)
        img = Image.open(io.BytesIO(img_data))
        tensor_img = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(tensor_img)
            pred = predict.argmax(dim=1).item()

            return {'Prediction': labels[pred]}

    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)