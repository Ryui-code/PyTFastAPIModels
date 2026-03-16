from fastapi import FastAPI
import uvicorn
from api import MNIST, CIFAR_10, Fashion_MNIST, Smartphones

app = FastAPI()

app.include_router(MNIST.mnist_router)
app.include_router(Fashion_MNIST.fashion_router)
app.include_router(CIFAR_10.cifar_router)
app.include_router(Smartphones.phones_router)

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)