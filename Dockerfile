FROM python:3.12

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

COPY . .

CMD uvicorn main:app --host 0.0.0.0 --port $PORT