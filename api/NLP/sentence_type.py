import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from fastapi import APIRouter, HTTPException
from googletrans import Translator
from schemas.pyd import SentenceTypeSchema

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SentenceTypeModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.linear = nn.Linear(128, 6)

    def forward(self, x):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        return self.linear(x[-1])

vocab = torch.load('models/sentence_type_vocab.pth', map_location=device, weights_only=False)

labels = torch.load('models/labels/sentence_type_labels.pth')
labels_pipeline = {index: label for index, label in enumerate(labels)}

model = SentenceTypeModel(len(vocab)).to(device)
model.load_state_dict(torch.load('models/sentence_type_model.pth', map_location=device))
model.eval()

tokenizer = get_tokenizer('basic_english')
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

translator = Translator()

sentence_type_router = APIRouter(prefix='/sentence_type', tags=['NLP'])

@sentence_type_router.post('/')
async def predict_text(schema: SentenceTypeSchema):
    text_data = (await translator.translate(schema.text, dest='en')).text

    tokens = text_pipeline(text_data)
    if not tokens:
        raise HTTPException(detail='Empty text or unknown symbols', status_code=400)

    tensor_text = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        predict = model(tensor_text)
        predict_index = torch.argmax(predict, dim=1).item()
        predict_label = labels_pipeline[predict_index]

        return {
            'text': schema.text,
            'label': predict_label
        }