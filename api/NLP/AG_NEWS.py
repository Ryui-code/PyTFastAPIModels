import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from fastapi import HTTPException, APIRouter
from schemas.pyd import AGNewsSchema
from langdetect import detect
from googletrans import Translator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AGNewsModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.linear = nn.Linear(128, 4)

    def forward(self, x):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        return self.linear(x[-1])

vocab = torch.load('models/ag_news_vocab.pth', map_location=device, weights_only=False)

labels = torch.load('models/labels/ag_news_labels.pth')
index_to_label = {index: label for index, label in enumerate(labels)}

model = AGNewsModel(len(vocab)).to(device)
model.load_state_dict(torch.load('models/ag_news_model.pth', map_location=device))
model.eval()

tokenizer = get_tokenizer('basic_english')
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

translator = Translator()

ag_news_router = APIRouter(prefix='/ag_news', tags=['NLP'])

@ag_news_router.post('/')
async def predict_text(schema: AGNewsSchema):
    try: user_lang = detect(schema.text)
    except: user_lang = 'en'

    translated_text = (await translator.translate(schema.text, dest='en')).text

    tokens = text_pipeline(translated_text)
    if not tokens:
        raise HTTPException(detail='Empty text or unknown symbols', status_code=400)

    x = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        predict = model(x)
        predict_index = torch.argmax(predict, dim=1).item()

        label_en = index_to_label[predict_index]
        translated_label = (await translator.translate(label_en, dest=user_lang)).text

        return {
            'text': schema.text,
            'label': translated_label
        }