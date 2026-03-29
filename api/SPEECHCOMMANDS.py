import torch
import torch.nn as nn
from torchaudio import transforms
import io
from fastapi import HTTPException, APIRouter, File, UploadFile
import torch.nn.functional as f
import soundfile as sf

class AudioLogic(nn.Module):
  def __init__(self):
    super().__init__()

    self.first = nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))
    )

    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 35)
    )

  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    return self.second(x)

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = AudioLogic().to(device)
model.eval()
model.load_state_dict(torch.load('models/speechcommands_model.pth', map_location=device))
labels = ['backward','bed','bird','cat','dog','down','eight','five',
          'follow','forward','four','go','happy','house','learn','left',
          'marvin','nine','no','off','on','one','right','seven','sheila',
          'six','stop','three','tree','two','up','visual','wow','yes','zero'
          ]

indx_to_lbl = {indx: lbl for indx, lbl in enumerate(labels)}

max_len = 100
def change_audio(waveform, sample_rate):
    waveform = torch.tensor(waveform).T

    if sample_rate != 16000:
        waveform = transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    spec = transform(waveform).squeeze(0)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]

    if spec.shape[1] < max_len:
        spec = f.pad(spec, (0, max_len - spec.shape[1]))
    return spec

speech_router = APIRouter(prefix='/speech', tags=['Audio To Text'])

@speech_router.post('/')
async def predict_sound(file: UploadFile = File(...)):
    try:
        sound_data = await file.read()
        if not sound_data:
            raise HTTPException(detail='Upload the file!', status_code=400)
        waveform, sample_rate = sf.read(io.BytesIO(sound_data), dtype='float32')
        spec = change_audio(waveform, sample_rate).unsqueeze(0).to(device)

        with torch.no_grad():
            predict = model(spec)
            pred_indx = torch.argmax(predict, dim=1).item()
            pred_lbl = indx_to_lbl[pred_indx]

            return {
                'index': pred_indx,
                'label': pred_lbl
            }

    except Exception as e:
        raise HTTPException(detail=str(e), status_code=500)