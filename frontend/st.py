import streamlit as st
from PIL import Image
import requests

st.sidebar.title("Task choice")

task = st.sidebar.radio(
    "Model type",
    ["Image to Text", "Audio to Text"]
)

if task == "Image to Text":
    image_model = st.sidebar.radio(
        "Image Models",
        ["CIFAR-10", "MNIST", "Fashion MNIST", "Smartphones", "CIFAR-100", "ImageScene"]
    )

    if image_model == "MNIST":
        api = 'http://127.0.0.1:8000/mnist/'

        st.title('MNIST Model')
        st.write('Upload the file and model will try to recognize it')

        uploaded_file = st.file_uploader('Choose image', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', width=200)

            if st.button('Predict'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Label: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')
    if image_model == 'Fashion MNIST':
        api = 'http://127.0.0.1:8000/fashion/'

        st.title('Fashion MNIST Model')
        st.write('Upload the file and model will try to recognize it')

        uploaded_file = st.file_uploader('Choose image', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', width=200)

            if st.button('Predict'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Label: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

    if image_model == 'CIFAR-10':
        api = 'http://127.0.0.1:8000/cifar-10'

        st.title('CIFAR-10 Model')
        st.write('Upload the file and model will try to recognize it')

        uploaded_file = st.file_uploader('Choose image', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', width=200)

            if st.button('Predict'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Label: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

    if image_model == 'Smartphones':
        api = 'http://127.0.0.1:8000/phones'

        st.title('Smartphones Model')
        st.write('Upload the file and model will try to recognize it')

        uploaded_file = st.file_uploader('Choose image', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', width=200)

            if st.button('Predict'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Label: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

    if image_model == 'CIFAR-100':
        api = 'http://127.0.0.1:8000/cifar-100'

        st.title('CIFAR-100 Model')
        st.write('Upload the file and model will try to recognize it')

        uploaded_file = st.file_uploader('Choose image', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', width=200)

            if st.button('Predict'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Label: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

    if image_model == 'ImageScene':
        api = 'http://127.0.0.1:8000/scene'

        st.title('Scene Model')
        st.write('Upload the file and model will try to recognize it')

        uploaded_file = st.file_uploader('Choose image', type=['png', 'jpeg', 'jpg'])

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption='Uploaded image', width=200)

            if st.button('Predict'):
                try:
                    files = {'file': (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f'Label: {result["Prediction"]}')
                    else:
                        st.error(f'Error {request.status_code}')
                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

elif task == "Audio to Text":
    audio_model = st.sidebar.radio(
        "Audio Models",
        ['SPEECHCOMMANDS', 'GTZAN', 'AudioMNIST', 'Urban']
    )

    if audio_model == 'SPEECHCOMMANDS':
        api = 'http://127.0.0.1:8000/speech'

        st.title(f"Audio Model {audio_model}")
        st.write("Upload the file and model will try to recognize it")

        audio_file = st.file_uploader("Choose audiofile", type=["wav", "mp3", "ogg"])

        if audio_file:
            st.audio(audio_file)

            if st.button("Predict"):
                try:
                    files = {"file": (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f"""
                        **Label:** {result['label']}  
                        **Index:** {result['index']}
                        """)

                    else:
                        st.error(f"Error: {request.status_code}")

                except requests.exceptions.RequestException:
                    st.error("Can not connect to the API")

    if audio_model == 'GTZAN':
        api = 'http://127.0.0.1:8000/gtzan'

        st.title(f'Audio Model {audio_model}')
        st.write('Upload the file and model will try to recognize it')

        audio_file = st.file_uploader('Choose audiofile', type=['mp3', 'wav', 'ogg'])

        if audio_file:
            st.audio(audio_file)

            if st.button('Predict'):
                try:
                    files = {'file': (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f"""
                        **Label:** {result['label']}  
                        **Index:** {result['index']}
                        """)

                    else:
                        st.error(f"Error: {request.status_code}")

                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

    if audio_model == 'AudioMNIST':
        api = 'http://127.0.0.1:8000/audio_mnist'

        st.title(f'Audio Model {audio_model}')
        st.write('Upload the file and model will try to recognize it')

        audio_file = st.file_uploader('Choose audiofile', type=['mp3', 'wav', 'ogg'])

        if audio_file:
            st.audio(audio_file)

            if st.button('Predict'):
                try:
                    files = {'file': (audio_file.name, audio_file.getvalue(), audio_file.type)}
                    request = requests.post(api, files=files)

                    if request.status_code == 200:
                        result = request.json()
                        st.success(f"""
                        **Label:** {result['label']}  
                        **Probability:** {result['probability']}
                        """)

                    else:
                        st.error(f"Error: {request.status_code}")

                except requests.exceptions.RequestException:
                    st.error('Can not connect to the server')

    if audio_model == 'Urban':
        api = "http://127.0.0.1:8000/urban"

        st.title(f"Audio Model {audio_model}")
        st.write("Upload the file and model will try to recognize it")

        mic_audio = st.audio_input("Record audio")

        file_audio = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])

        if st.button("Predict"):
            try:
                if mic_audio:
                    files = {"file": (mic_audio.name, mic_audio.getvalue(), mic_audio.type)}
                    request = requests.post(api, files=files)

                elif file_audio:
                    files = {"file": (file_audio.name, file_audio.getvalue(), file_audio.type)}
                    request = requests.post(api, files=files)

                else:
                    st.warning("Upload the file")
                    request = None

                if request and request.status_code == 200:
                    result = request.json()
                    st.success(f"""
                    **Label:** {result['label']}  
                    **Index:** {result['index']}
                    """)
                elif request:
                    st.error(f"Error: {request.status_code}")

            except requests.exceptions.RequestException:
                st.error("Cannot connect to the server")
