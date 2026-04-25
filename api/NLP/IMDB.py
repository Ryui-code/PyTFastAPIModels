import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from fastapi import APIRouter, HTTPException
from schemas.pyd import IMDBSchema
# from langdetect import detect
from googletrans import Translator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentimentalModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.fc(hidden[-1])

vocab = torch.load('models/imdb_vocab.pth', map_location=device, weights_only=False)

model = SentimentalModel(len(vocab)).to(device)
model.load_state_dict(torch.load('models/imdb_model.pth', map_location=device))
model.eval()

tokenizer = get_tokenizer('basic_english')
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

imdb_router = APIRouter(prefix='/imdb')

translator = Translator()

@imdb_router.post('/', tags=['NLP'])
async def predict_text(schema: IMDBSchema):
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
        predict_label = 'positive' if predict_index == 1 else 'negative'

        return {
            'text': schema.text,
            'label': predict_label
        }