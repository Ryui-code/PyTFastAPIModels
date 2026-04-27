import torch
import torch.nn as nn
from torchtext.data import get_tokenizer
from fastapi import HTTPException, APIRouter
from googletrans import Translator
from schemas.pyd import YelpReviewSchema

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class YelpReviewModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, 64)
        self.lstm = nn.LSTM(64, 128, batch_first=True)
        self.linear = nn.Linear(128, 5)

    def forward(self, x):
        x = self.embedding(x)
        _, (x, _) = self.lstm(x)
        return self.linear(x[-1])

vocab = torch.load('models/yelp_review_vocab.pth', weights_only=False)

model = YelpReviewModel(len(vocab)).to(device)
model.load_state_dict(torch.load('models/yelp_review_model.pth', map_location=device))
model.eval()

labels = ['⭐ 1 Star', '⭐ 2 Stars', '⭐ 3 Stars', '⭐ 4 Stars', '⭐ 5 Stars']
index_to_label = {index: label for index, label in enumerate(labels)}

tokenizer = get_tokenizer('basic_english')
def text_pipeline(text):
    return [vocab[token] for token in tokenizer(text)]

translator = Translator()

yelp_review_router = APIRouter(prefix='/yelp_review', tags=['NLP'])

@yelp_review_router.post('/')
async def predict_text(schema: YelpReviewSchema):
    text_data = (await translator.translate(schema.text, dest='en')).text

    tokens = text_pipeline(text_data)
    if not tokens:
        raise HTTPException(detail='Emtpy text or unknown symbols', status_code=400)

    tensor_text = torch.tensor(tokens, dtype=torch.int64).unsqueeze(0).to(device)

    with torch.no_grad():
        predict = model(tensor_text)
        predict_index = torch.argmax(predict, dim=1).item()
        predict_label = index_to_label[predict_index]

        return {
            'text': schema.text,
            'label': predict_label
        }