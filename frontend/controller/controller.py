from model.RecordingToNotesConverter import *
import matplotlib.pyplot as plt
import librosa

y, sr = librosa.load("examples/frere-jacques.mp3")
oenv = librosa.onset.onset_strength(y=y, sr=sr)
times = librosa.times_like(oenv)

loader = Loader()
loader.initialize_audio_file("examples/frere-jacques.mp3")
model = Model("examples/promenade_trumpet.wav","C4","C6")


notes, bpm = model.wave_to_notes()
model.generate_music_sheet(notes, bpm)