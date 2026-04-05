from fastapi import FastAPI
import uvicorn
from api import MNIST, CIFAR_10, FashionMNIST, Smartphones, CIFAR_100, ImageScene, SPEECHCOMMANDS, GTZAN, audio_mnist, urban

app = FastAPI()

app.include_router(MNIST.mnist_router)
app.include_router(Fashion_MNIST.fashion_router)
app.include_router(CIFAR_10.cifar_10_router)
app.include_router(Smartphones.phones_router)
app.include_router(CIFAR_100.cifar_100_router)
app.include_router(ImageScene.scene_router)
app.include_router(GTZAN.gtzan_router)
app.include_router(SPEECHCOMMANDS.speech_router)
app.include_router(audio_mnist.audio_mnist_router)
app.include_router(urban.urban_router)

if __name__ == '__main__':
    uvicorn.run('main:app', reload=True)