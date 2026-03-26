import streamlit as st
from PIL import Image
import requests

st.sidebar.title("Выбор задачи")

task = st.sidebar.radio(
    "Тип модели",
    ["Image to Text", "Audio to Text"]
)

if task == "Image to Text":
    image_model = st.sidebar.radio(
        "Модели для изображений",
        ["CIFAR-10", "MNIST", "Fashion MNIST", "Smartphones", "CIFAR-100", "ImageScene"]
    )

    if image_model == "MNIST":
        api = 'http://127.0.0.1:8000/mnist/'

        st.title('MNIST Model')
        st.write('Загрузите изображение с цифрой и модель попробует распознать')

        uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', width=200)

            if st.button('Определить'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f"Модель думает, что это цифра: {result['Prediction']}")
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Cannot connect to the API')

    if image_model == 'Fashion MNIST':
        api = 'http://127.0.0.1:8000/fashion/'

        st.title('Fashion MNIST Model')
        st.write('Загрузите изображение и модель попробует распознать')

        uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', width=200)

            if st.button('Определить'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Модель думает, что это: {result['Prediction']}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Cannot connect to the API')

    if image_model == 'CIFAR-10':
        api = 'http://127.0.0.1:8000/cifar-10'

        st.title('CIFAR-10 Model')
        st.write('Загрузите изображение и модель попробует распознать')

        uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', width=200)

            if st.button('Определить'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Модель думает что это: {result['Prediction']}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Cannot connect to the API')

    if image_model == 'Smartphones':
        api = 'http://127.0.0.1:8000/phones'

        st.title('Smartphones Model')
        st.write('Загрузите изображение бренда одного из брендов')

        uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpg', 'jpeg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', width=200)

            if st.button('Определить'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Модель думает что это: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Cannot connect to the API')

    if image_model == 'CIFAR-100':
        api = 'http://127.0.0.1:8000/cifar-100'

        st.title('CIFAR-100 Model')
        st.write('Загрузите изображение и модель попробует ее распознать')

        uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', width=200)

            if st.button('Определить'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Модель думает что это: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Cannot connect to the server')

    if image_model == 'ImageScene':
        api = 'http://127.0.0.1:8000/scene'

        st.title('Scene Model')
        st.write('Загрузите изображение и модель попробует ее распознать')

        uploaded_file = st.file_uploader('Выберите изображение', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Загруженное изображение', width=200)

            if st.button('Определить'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Модель думает что это: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Cannot connect to the server')

elif task == "Audio to Text":
    audio_model = st.sidebar.radio(
        "Модели для аудио",
        ['SPEECHCOMMANDS']
    )

    if audio_model == 'SPEECHCOMMANDS':
        api = 'http://127.0.0.1:8000/speech'

        st.title(f"Audio Model: {audio_model}")
        st.write("Загрузите аудиофайл, и модель преобразует его в текст")

        audio_file = st.file_uploader("Выберите аудио", type=["wav", "mp3", "ogg"])

        if audio_file:
            st.audio(audio_file)

            if st.button("Распознать"):
                try:
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    response = requests.post(api, files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.write(f"Индекс класса: {result['index']}")
                        st.success(f"Предсказанная команда: **{result['label']}**")

                    else:
                        st.error(f"Error: {response.status_code}")

                except requests.exceptions.RequestException:
                    st.error("Can not connect to the API")