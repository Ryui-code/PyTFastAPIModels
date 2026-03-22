import streamlit as st
from PIL import Image
import requests
with st.sidebar:
    models = st.radio('Models', ['CIFAR-10', 'MNIST', 'Fashion MNIST', 'Smartphones', 'CIFAR-100'])

if models == 'MNIST':
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
                    st.success(f'Модель думает, что это цифра: {result['Prediction']}')
                else:
                    st.error(f'Error {request.status_code}')
            except requests.exceptions.RequestException:
                st.error('Cannot connect to the API')

if models == 'Fashion MNIST':
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

if models == 'CIFAR-10':
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

if models == 'Smartphones':
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

if models == 'CIFAR-100':
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