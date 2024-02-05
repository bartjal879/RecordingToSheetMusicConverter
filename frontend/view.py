import io
import os
import zipfile
import streamlit as st
import pandas as pd
import numpy as np
import requests
from plotly.offline import iplot
import plotly.graph_objs as go
import plotly.express as px
from pandas.io.json._normalize import json_normalize
import requests

st.write("""
# Aplikacja do konwersji nagrania audio na zapis nutowy ✌
""")
st.write("Konwersja nagrania audio na zapis nutowy jest możliwa dzięki wykorzystaniu algorytmów przetwarzania sygnałów dźwiękowych. W tym przypadku do zagadnienia automatycznej transkrypcji muzyki wykorzystano algorytmy Viterbi'ego oraz pYin.")
st.write("""
# Recording to sheet music conversion application ✌
""")

st.write('Conversion of audio recordings into musical notation is possible thanks to the utilization of audio signal processing algorithms. In this case, Viterbi and pYin algorithms have been used for the task of automatic music transcription.')
# st.set_option('deprecation.showfileUploaderEncoding', False)

    
st.sidebar.header('Ustawienia konwersji')
st.sidebar.subheader('Przykładowe pliki audio')
files = os.listdir('examples')
selected_file = st.sidebar.selectbox('Lista przykładowych plików audio do konwersji', files, index=None, placeholder="Wybierz przyładowy plik audio")
st.sidebar.subheader('Plik użytkownika do konwersji')
uploaded_file = st.sidebar.file_uploader("Prześlij plik, który chcesz przekonwertować", type=["mp3", "wav", "flac"])
if uploaded_file is not None:
    selected_file = None
    st.sidebar.audio(uploaded_file)
elif selected_file is not None:
    uploaded_file = None
    st.sidebar.audio(f'examples/{selected_file}')
else:
    st.write("# ⬅ Wprowadź dane wejściowe na pasku bocznym, aby zobaczyć wyniki konwersji.")
    
# if (selected_file == 'frere-jacques.mp3'):
st.sidebar.subheader('Metody konwersji i ich parametry')

conversion_method = st.sidebar.radio(
    "Której motody konwersji chcesz użyć?",[0, 1],
    captions = ["Algorytm pYin, wykrywanie początków dźwięków i predykcja czasu ich trwania na jej podstawie", 
                "Algorytm pYin, HMM z wykorzystaniem algorytmu Viterbiego" ])



if conversion_method == 1: 
    sustain_slider = st.sidebar.slider('Prawdopodobienstwo przetrzymania nuty() (%)', min_value=0, max_value=100, value=70)
    silence_slider = st.sidebar.slider('Prawdopodobienstwo ciszy (%)', min_value=0, max_value=100, value=70)
    ratio_slider = st.sidebar.slider('Stosunek pr. przedluzenia nuty do powtorzenia () (%)', min_value=0, max_value=100, value=80)
    onset_slider = st.sidebar.slider('Prawdopodobienstwo onsetu (%)', min_value=0, max_value=100, value=90)
    f0_slider = st.sidebar.slider('Prawdopodobienstwo estymacji czestotliwosci (%)', min_value=0, max_value=100, value=90)

notes = ['C', 'D', 'E', 'F', 'G', 'A', 'H']
octaves = list(range(8))

notes = [note + str(octave) for octave in octaves for note in notes ]

start_note, end_note = st.sidebar.select_slider(
    'Wybierz zakres wysokości dźwięków konwertowanego utworu',
    options=notes,
    value=('C4', 'C6'))

# selected_onset_method = st.sidebar.selectbox('Lista metod detekcji początków dźwięków w aplikacji.', , index=None, placeholder="Wybierz przyładowy plik audio")
selected_onset_method = st.sidebar.radio('Lista metod detekcji początków dźwięków w aplikacji.', ["Średnia (Mel)", "Mediana (niestandardowy Mel)", "Średnia (CQT)"], index=0, 
    captions = ["Podstawowa metoda detekcji początków dźwięków oparta o obliczanie średniej wartości algorytmu MFCCs (Mel-frequency cepstral coefficients)", 
                "Wykrycie początków dźwięków za pomocą mediany wartości algorytmu MFCCs. Przydatna przy wyższych aplitudach mocy sygnału dźwiękowego",
                "Wykrycie początków dźwięków za pomocą średniej wartości algorytmu Constant-Q Transform. Przydatna przy niższych aplitudach mocy sygnałuy dźwiękowego" ])

if selected_onset_method == "Średnia (Mel)":
    onset_method = 0
elif selected_onset_method == "Mediana (niestandardowy Mel)":
    onset_method = 1
    custom_mel_slider = st.sidebar.slider('Liczba mel:', min_value=0, max_value=1000, value=256)
else:
    onset_method = 2

def convert():
    st.write('# Wyniki konwersji')
    url = 'http://127.0.0.1:8000/'
    file = {'file': open(f"examples/{selected_file}",'rb') if selected_file is not None else uploaded_file}

    response = requests.post(f"{url}audio_converter/", files=file)

    if response.status_code != 200:
       st.write("# Nastąpił błąd podczas przesyłania pliku audio.")
   
    params = {
            'start_note': start_note, 'end_note': end_note, 'conversion_method': conversion_method,
            'onset_method': onset_method, 
            } 
    if conversion_method == 1:
        params.update({
            'sustain_probability': sustain_slider/100, 'silence_probability': silence_slider/100,
            'ratio': ratio_slider/100, 'onset_probability': onset_slider/100, 'f0_probability': f0_slider/100
            })
    if onset_method == 1:
        params.update({'custom_mel': custom_mel_slider})

    # podstawowa jednostka metrum
    response = requests.get(f"{url}audio_converter/", params=params)
    if response.status_code == 200:
        zip_io = io.BytesIO(response.content)

        # Variables to store the file data
        pdf_data = None
        png_data = None
        ly_data = None

        # Open the zip file
        with zipfile.ZipFile(zip_io, 'r') as zipf:
            # Loop over each file in the archive
            for filename in zipf.namelist():
                # Open each file
                with zipf.open(filename) as f:
                    # Read the file as bytes
                    file_data = f.read()

                    # Check the file extension and store the data in the appropriate variable
                    _, ext = os.path.splitext(filename)
                    if ext.lower() == '.pdf':
                        pdf_data = file_data
                    elif ext.lower() == '.png':
                        png_data = file_data
                    elif ext.lower() == '.ly':
                        ly_data = file_data

        # Display the PNG image
        st.write("Nagranie audio zostało przekonwertowane na zapis nutowy. Poniżej znajdziesz podgląd wyników konwersji oraz linki do pobrania plików: PDF zawierającego zapis nutowy, LY zawierającego zapis nutowy w formacie aplikacji do edycji nut LilyPond, oraz obu plików wraz z plikiem podglądu PNG.")
        col1, col2 = st.columns([2,1])
        col1.image(png_data)
        col2.download_button(
               label="Pobierz plik PDF z zapisem nutowym",
               data=pdf_data,
               file_name="result.pdf",
               mime="application/pdf",
           )        
        col2.download_button(
               label="Pobierz plik LY do edycji w aplikacji obsługującej format LilyPond",
               data=ly_data,
               file_name="result.ly",
               mime="application/ly",
           )
        col2.download_button(
               label="Pobierz plik ZIP z plikami PDF i LY",
               data=response.content,
               file_name="results.zip",
               mime="application/zip",
           )                
    else:
        st.write("# Nastąpił błąd podczas konwersji pliku audio.")

    

if uploaded_file is not None or selected_file is not None:
    
   
       
    convert_button = st.sidebar.button('Przekonwertuj plik')

    if convert_button:
        convert()
    # if result=='positive':
    #     st.write("""# Great Work there! You got a Positive Review 😃""")
    # elif result=='negative':
    #     st.write("""# Try improving your product! You got a Negative Review 😔""")
    # else:
    #     st.write("""# Good Work there, but there's room for improvement! You got a Neutral Review 😶""")
    
    # Then you can check the response
    # if response.status_code == 200:
    #     print('Audio file uploaded successfully')
    # else:
    #     print('Failed to upload audio file')
    # input_df = pd.read_csv(uploaded_file)
    # for i in range(input_df.shape[0]):
    #     url = 'http://127.0.0.1:8000/convert/?text='+str(input_df.iloc[i])
    #     r = requests.get(url)
    #     result = r.json()["text_sentiment"]
    #     if result=='positive':
    #         count_positive+=1
    #     elif result=='negative':
    #         count_negative+=1
    #     else:
    #         count_neutral+=1 

    # x = ["Positive", "Negative", "Neutral"]
    # y = [count_positive, count_negative, count_neutral]

    # if count_positive>count_negative:
    #     st.write("""# Great Work there! Majority of people liked your product 😃""")
    # elif count_negative>count_positive:
    #     st.write("""# Try improving your product! Majority of people didn't find your product upto the mark 😔""")
    # else:
    #     st.write("""# Good Work there, but there's room for improvement! Majority of people have neutral reactions to your product 😶""")
        
    # layout = go.Layout(
    #     title = 'Multiple Reviews Analysis',
    #     xaxis = dict(title = 'Category'),
    #     yaxis = dict(title = 'Number of reviews'),)
    
    # fig.update_layout(dict1 = layout, overwrite = True)
    # fig.add_trace(go.Bar(name = 'Multi Reviews', x = x, y = y))
    # st.plotly_chart(fig, use_container_width=True) 
   



st.sidebar.image('logo.jpg')
st.sidebar.subheader("""Projekt dyplomowy realizowany na końcowym semestrze kierunku Informatyka na Politechnice Śląskiej, Wydział Automatyki, Elektorniki i Informatyki, Katowice, Polska.""")
st.sidebar.subheader("""Final year project at Computer Science course in Silesian University of Technology, Faculty of Automation, Electronics and Computer Science, Katowice, Poland.""")