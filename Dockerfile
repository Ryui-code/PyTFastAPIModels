FROM python:3.12

WORKDIR /app

COPY req.txt /app/
RUN pip install gunicorn
RUN pip install setuptools
RUN pip install --upgrade pip && \
    pip install -r req.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]