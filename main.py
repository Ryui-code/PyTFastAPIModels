from fastapi import FastAPI
# import uvicorn
from api.CV import CIFAR_10, CIFAR_100, FashionMNIST, ImageScene, MNIST, Smartphones, TRASH
from api.SR import audio_mnist, GTZAN, SPEECHCOMMANDS, urban, environmental_sound, emotional_speech
from api.NLP import IMDB, AG_News, go_emotions, yelp_review

app = FastAPI()

app.include_router(MNIST.mnist_router)
app.include_router(FashionMNIST.fashion_router)
app.include_router(CIFAR_10.cifar_10_router)
app.include_router(Smartphones.phones_router)
app.include_router(CIFAR_100.cifar_100_router)
app.include_router(ImageScene.scene_router)
app.include_router(GTZAN.gtzan_router)
app.include_router(SPEECHCOMMANDS.speech_router)
app.include_router(audio_mnist.audio_mnist_router)
app.include_router(urban.urban_router)
app.include_router(TRASH.trash_router)
app.include_router(environmental_sound.environmental_sound_router)
app.include_router(emotional_speech.emotional_speech_router)
app.include_router(IMDB.imdb_router)
app.include_router(AG_News.ag_news_router)
app.include_router(go_emotions.go_emotions_router)
app.include_router(yelp_review.yelp_review_router)

# if __name__ == '__main__':
#     uvicorn.run('main:app', reload=True)