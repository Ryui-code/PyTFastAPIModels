import streamlit as st
from PIL import Image
import requests

st.sidebar.title("Task choice")

task = st.sidebar.radio(
    "Model type",
    ["Computer Vision", "Speech Recognition", 'Natural Language Processing']
)

if task == "Computer Vision":
    cv_model = st.sidebar.radio(
        "CV Models",
        ["CIFAR-10", "MNIST", "Fashion MNIST", "Smartphones", "CIFAR-100", "ImageScene", 'TRASH']
    )

    if cv_model == "MNIST":
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

    if cv_model == 'Fashion MNIST':
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

    if cv_model == 'CIFAR-10':
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

    if cv_model == 'Smartphones':
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

    if cv_model == 'CIFAR-100':
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

    if cv_model == 'ImageScene':
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

    if cv_model == 'TRASH':
        api = 'http://127.0.0.1:8000/trash'

        st.title('Trash Model')
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

elif task == "Speech Recognition":
    sr_model = st.sidebar.radio(
        "SR Models",
        ['SPEECHCOMMANDS', 'GTZAN', 'AudioMNIST', 'Urban', 'Environmental Sound', 'Emotional Speech']
    )

    if sr_model == 'SPEECHCOMMANDS':
        api = 'http://127.0.0.1:8000/speech'

        st.title(f"Audio Model {sr_model}")
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

    if sr_model == 'GTZAN':
        api = 'http://127.0.0.1:8000/gtzan'

        st.title(f'Audio Model {sr_model}')
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

    if sr_model == 'AudioMNIST':
        api = 'http://127.0.0.1:8000/audio_mnist'

        st.title(f'Audio Model {sr_model}')
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

    if sr_model == 'Urban':
        api = "http://127.0.0.1:8000/urban"

        st.title(f"Audio Model {sr_model}")
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

    if sr_model == 'Environmental Sound':
        api = "http://127.0.0.1:8000/environmental_sound"

        st.title(f"Audio Model {sr_model}")
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

    if sr_model == 'Emotional Speech':
        api = "http://127.0.0.1:8000/emotional_speech"

        st.title(f"Audio Model {sr_model}")
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

if task == "Natural Language Processing":
    nlp_model = st.sidebar.radio(
        "NLP Models",
        ['IMDB', 'AG News']
    )

    if nlp_model == 'IMDB':
        api = 'http://127.0.0.1:8000/imdb'

        st.title(f'Model {nlp_model}')

        text = st.text_area('Review Text', placeholder='Enter your movie review here...')
        sentiment_data = {
            'text': text
        }

        if st.button('Predict'):
            try:
                request = requests.post(api, json=sentiment_data, timeout=10)
                if request.status_code == 200:
                    result = request.json()
                    st.success(f"""
                    **Text:** {result['text']}  
                    **Label:** {result['label']}
                    """)
                else:
                    st.error(f'Error: {request.status_code}')
            except requests.exceptions.RequestException:
                st.error('Can not connect to the API')

    if nlp_model == 'AG News':
        api = 'http://127.0.0.1:8000/ag_news'

        st.title(f'Model {nlp_model}')

        text = st.text_area('Text', placeholder='Write text and model will try to recognize it')
        sentiment_data = {
            'text': text
        }

        if st.button('Predict'):
            try:
                request = requests.post(api, json=sentiment_data, timeout=10)
                if request.status_code == 200:
                    result = request.json()
                    st.success(f"""
                    **Text:** {result['text']}
                    
                    **Label:** {result['label']}
                    """)
                else:
                    st.error(f'Error: {request.status_code}')
            except requests.exceptions.RequestException:
                st.error('Can not connect to the API')
