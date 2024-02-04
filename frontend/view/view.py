import streamlit as st
from model.RecordingToNotesConverter import *

def main():
    # Create a list of file names
    files = ['c_maj_scale.wav', 'frere-jacques.mp3', 'trumpet.wav']

    # Create a list of display names
    display_names = ['Scale', 'Frere Jaques', 'Trumpet']
    
    # Create a dictionary to map display names to file names
    name_map = dict(zip(display_names, files))
    
    
# Create two sliders
    

    check = st.checkbox('Priorytezacja wykrytych onsetow')

    

    # Create lists of notes and octaves
    notes = ['C', 'D', 'E', 'F', 'G', 'A', 'H/B']
    octaves = list(range(11))
    
    # Create two rows of two columns each
    row1 = st.columns(2)
    row2 = st.columns(2)
    
    # Create select boxes for the start note and octave
    start_note = row1[0].selectbox('Select a start note', notes)
    start_octave = row1[1].selectbox('Select a start octave', octaves)
    
    # Create select boxes for the end note and octave
    end_note = row2[0].selectbox('Select an end note', notes)
    end_octave = row2[1].selectbox('Select an end octave', octaves)
    
    st.write(f'You selected the note range: {start_note}{start_octave} to {end_note}{end_octave}')
    # Create a container
    container = st.container()
    
    # Inside the container, create a selectbox and a file uploader
    with container:
        selected_display_name = st.selectbox('Select a file', display_names)

        # Get the selected file name
        selected_file = name_map[selected_display_name]
        
        st.write(f'You selected: {selected_display_name}, which corresponds to the file: {selected_file}')
        uploaded_file = st.file_uploader('Upload a file')
    
        # Display the values of the sliders and the selectbox
        st.write(f'Slider 1 value: {probabliity_slider}')
        st.write(f'Slider 2 value: {slider2}')
        # st.write(f'Selected option: {selectbox}')
    
        # If a file is uploaded, display a message
        if uploaded_file is not None:
            st.write('A file has been uploaded.')
        
        calculate = st.button('Calculate Results')
        # Create a "Calculate Results" button
        if(calculate):
            loader = Loader()
            model = Model(f"{start_note}{start_octave}", f"{end_note}{end_octave}", check)
            loader.initialize_audio_file(selected_file)
            notes, bpm = model.wave_to_notes(loader.get_audio_data(), loader.get_sampling_rate(), p_stay_note=probabliity_slider/100, p_stay_silence=slider2/100, ratio=ratio_slider/100,
                                             note_min=f"{start_note}{start_octave}", note_max=f"{end_note}{end_octave}", pitch_acc=f0_slider/100, onset_acc=onset_slider/100 )
            model.generate_music_sheet(notes, bpm)

if __name__ == "__main__":
    main()

# v.initialize_view()
# play(guitar, bpm=100, instrument=25)
# convert_to_wav(, "C:/Users/JaBartek/Desktop/Polsl/INZ/RecordingToNotesConverter/RecordingToNotesConverter/RecordingToNotesConverter/examples/frere-jacques.wav")

# conv.initialize_audio_file("examples/trumpet.wav")

# conv = Model()
# notes, bpm = conv.wave_to_notes(loader.get_audio_data(), loader.get_sampling_rate())
# #create a string from the list of string elements

# conv.generate_music_sheet(notes, bpm)