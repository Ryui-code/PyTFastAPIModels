import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from fastapi import HTTPException, APIRouter
from googletrans import Translator
from langdetect import detect
from schemas.pyd import GoEmotionsSchema

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GoEmotionsModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.linear = nn.Linear(128, 28)

    def forward(self, x):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        return self.linear(x[-1])

vocab = torch.load('models/go_emotions_vocab.pth', map_location=device, weights_only=False)

labels = torch.load('models/labels/go_emotions_labels.pth')
index_to_label = {index: label for index, label in enumerate(labels)}

model = GoEmotionsModel(len(vocab)).to(device)
model.load_state_dict(torch.load('models/go_emotions_model.pth', map_location=device))
model.eval()

tokenizer = get_tokenizer('basic_english')
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

translator = Translator()

go_emotions_router = APIRouter(prefix='/go_emotions', tags=['NLP'])

@go_emotions_router.post('/')
@go_emotions_router.post('/')
async def predict_text(schema: GoEmotionsSchema):
    # try: user_lang = detect(schema.text)
    # except: user_lang = 'en'

    translated_text = (await translator.translate(schema.text)).text

    tokens = text_pipeline(translated_text)
    if not tokens:
        raise HTTPException(detail='Empty text or unknown symbols', status_code=400)

    x = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        predict = model(x)
        predict_index = torch.argmax(predict, dim=1).item()
        predict_label = index_to_label[predict_index]

        return {
            'text': schema.text,
            'label': predict_label
        }