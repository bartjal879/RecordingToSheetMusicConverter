import array
from django.db import models
from ast import parse
from email.mime import audio
import fractions
from math import floor, isnan
from collections import Counter
import math
import librosa
import subprocess
import scipy.signal as signal
import abjad
import lilypond
import matplotlib.pyplot as plt
from itertools import combinations
from music21 import note, stream, duration, meter, environment, converter
import soundfile as sf
import numpy as np
from enum import Enum
import os
import abjad
import lilypond
import subprocess

# Create your models here.

class Model:
    
    class State(Enum):
        SILENCE = 0
        ONSET = 1
        SUSTAIN = 2

    def __init__(self):
        self.midi_min = 0
        self.midi_max = 0
        self.n_notes = 0
        self.fmin = 0
        self.fmax = 0
        self.__audio_data = []
        self.__sampling_rate = 0
        self.simple_advanced_algorithm = False
        self.onset_method_conditional_parameter = 1
        self.__f0_ = []
        self.__voiced_flags = []
        self.__onsets = []

        
        
    def initialize_audio_data(self, file_path: str):
        self.__audio_data, self.__sampling_rate = librosa.load(file_path)
        print("Succesfully initialiazed audio data")

        
    def initialize_audio_parameters(self, note_min: str, note_max: str, onset_method: int = 1, frame_length: int = 2048, hop_length: int = 512 ): 
        self.midi_min = librosa.note_to_midi(note_min) 
        self.midi_max = librosa.note_to_midi(note_max) 
        self.n_notes = self.midi_max - self.midi_min + 1 
        self.fmin = librosa.note_to_hz(note_min) 
        self.fmax = librosa.note_to_hz(note_max)
        # self.conversion_method = conversion_method
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.onset_method = onset_method
        print("onset method",onset_method)
        print("Succesfully initialiazed audio parameters")
        
    def __calculate_transition_matrix(self, p_stay_note: float, p_stay_silence: float, ratio: float) -> np.array:
        # p_l = prawdopodobieństwo wykrycia ciszy.
        p_l = (1 - p_stay_silence) / self.n_notes

        #P_ll = prawodpodobieństwo pozostania w stanie sustain.
        p_ll = (1 - p_stay_note) / (self.n_notes)
        
        # Macierz przejść:
        transmat = np.zeros((2 * self.n_notes + 1, 2 * self.n_notes + 1))

        # Stan 0: cisza
        transmat[0, 0] = p_stay_silence
        for i in range(self.n_notes):
            transmat[0, (i * 2) + 1] = p_l

        # Stany nieparzyste - 1, 3, 5... = początki dźwięków
        for i in range(self.n_notes):
            transmat[(i * 2) + 1, (i * 2) + 2] = 1

        # Stany parzyste - 2, 4, 6... = trwanie dźwięków
        for i in range(self.n_notes):
            transmat[(i * 2) + 2, 0] = p_ll
            transmat[(i * 2) + 2, (i * 2) + 2] = p_stay_note*ratio
            for j in range(self.n_notes):
                transmat[(i * 2) + 2, (j * 2) + 1] = p_ll
            transmat[(i * 2) + 2, (i * 2) + 1] = p_stay_note*(1-ratio)

        return transmat    
    def __calculate_onsets_and_f0_candidates(self, audio_signal: np.array, srate: int, frame_length: int = 2048, hop_length: int = 512) -> (np.array, np.array):
         # kalkulacja kandydatów na częstotliowści podstawowe i flagi "dźwięczności" dla każdej ramki
        pitch, self.__voiced_flags, _ = librosa.pyin(
            y=audio_signal, fmin=self.fmin * 0.9, fmax=self.fmax * 1.1,
            sr=srate, frame_length=frame_length, win_length=int(frame_length / 2),
            hop_length=hop_length)
        tuning = librosa.pitch_tuning(pitch)
        self.__f0_ = np.round(librosa.hz_to_midi(pitch - tuning)).astype(int)
        print("calculating onsets")
        if self.onset_method == 0:
            o_env = librosa.onset.onset_strength(y=self.__audio_data, sr=self.__sampling_rate)
        elif self.onset_method == 1:
             o_env = librosa.onset.onset_strength(y=self.__audio_data, sr=self.__sampling_rate,  aggregate=np.median, fmax=self.fmax * 1.5, n_mels=256)        
        # elif self.onset_method_conditional_parameter == 2:
        #      o_env = librosa.onset.onset_strength(y=self.__audio_data, sr=self.__sampling_rate, feature=librosa.feature.cq)
        else:
            C = np.abs(librosa.cqt(y=self.__audio_data, sr=self.__sampling_rate))
            o_env = librosa.onset.onset_strength(sr=self.__sampling_rate, S=librosa.amplitude_to_db(C, ref=np.max))
        
        self.__onsets = librosa.onset.onset_detect(
            y=audio_signal, sr=srate, onset_envelope=o_env,
            hop_length=hop_length, backtrack=False)
        return self.__onsets, self.__f0_
    
    def __calculate_prior_probabilities(self, audio_signal: np.array, srate: int, frame_length: int = 2048, hop_length: int = 512, 
                          pitch_acc: float = 0.9, voiced_acc: float = 0.9, onset_acc: float = 0.9, spread: float = 0.2) -> np.array:
        # o_env = librosa.onset.onset_strength(y=audio_signal, sr=srate)
        # times = librosa.times_like(o_env, sr=srate)
        # onset_frames = librosa.onset.onset_detect(onset_envelope=o_env, sr=srate)
        # D = np.abs(librosa.stft(audio_signal))
        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
        #                          x_axis='time', y_axis='log', ax=ax[0])
        # ax[0].set(title='Power spectrogram')
        # ax[0].label_outer()
        # ax[1].plot(times, o_env, label='Onset strength')
        # ax[1].vlines(times[onset_frames], 0, o_env.max(), color='r', alpha=0.9,
        #            linestyle='--', label='Onsets')
        # ax[1].legend()
        

        
        priors = np.ones((self.n_notes * 2 + 1, len(self.__f0_)))

        for n_frame in range(len(self.__f0_)):
            if not self.__voiced_flags[n_frame]:
                priors[0, n_frame] = voiced_acc
            else:
                priors[0, n_frame] = 1 - voiced_acc

            for j in range(self.n_notes):
                if n_frame in self.__onsets:
                    priors[(j * 2) + 1, n_frame] = onset_acc
                else:
                    priors[(j * 2) + 1, n_frame] = 1 - onset_acc

                if j + self.midi_min == self.__f0_[n_frame]:
                    priors[(j * 2) + 2, n_frame] = pitch_acc
                    
                elif np.abs(j + self.midi_min - self.__f0_[n_frame]) == 1:
                    priors[(j * 2) + 2, n_frame] = pitch_acc * spread

                else:
                    priors[(j * 2) + 2, n_frame] = 1 - pitch_acc
                        
        return priors
    
    def __onsets_to_notesroll(self, hop_time: float) -> list:
        # Możliwe typy stanów:
        silence = 0
        onset = 1
        sustain = 2
        
            
        my_state = silence
        output = []
    
        last_onset = 0
        last_offset = 0
        last_midis = []
        for i, _ in enumerate(self.__f0_):
            if my_state == silence:
                if i in self.__onsets:    
                    # Znaleziony onset.
                    last_midis = []
                    last_onset = i * hop_time
                    last_midis.append(self.__f0_[i])
                    # last_note = librosa.midi_to_note(last_midis)
                    my_state = onset
    
            elif my_state == onset:
                 if i not in self.__onsets:
                    # Znaleziona faza brzmienia dźwięku.    
                    my_state = sustain
                    last_midis.append(self.__f0_[i])
    
            elif my_state == sustain:
                if i in self.__onsets:
                    # Znaleziony onset.
                    # Dokończenie poprzedniej nuty.

                    last_offset = i * hop_time
                    last_midi = Counter(last_midis).most_common(1)[0][0]
                    last_note = librosa.midi_to_note(last_midi)
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)

                    # Rozpoczęcie nowej nuty.

                    last_midis = []
                    last_onset = i * hop_time
                    last_midis.append(self.__f0_[i])
                    # last_note = librosa.midi_to_note(last_midis)
                    my_state = onset 
                    
                elif not (self.__voiced_flags[i] or self.__voiced_flags[i+1]):
                    # Wykryta cisza. 
                    # Dokończenie poprzedniej nuty.

                    last_offset = i * hop_time
                    last_midi = Counter(last_midis).most_common(1)[0][0]
                    last_note = librosa.midi_to_note(last_midi)
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)
                    my_state = silence
                else:
                    last_midis.append(self.__f0_[i])
        
        # for i, sequence in enumerate(output):
        #     print(i, sequence)
        # print(self.__onsets)
        return output

    
    
    def __states_to_notesroll(self, states: list, hop_time: float) -> list:
        # midi_min = librosa.note_to_midi(note_min)
    
        states_ = np.hstack((states, np.zeros(1)))
    
        # Możliwe typy stanów:
        silence = 0
        onset = 1
        sustain = 2
        
        my_state = silence
        output = []
    
        last_onset = 0
        last_offset = 0
        last_midi = 0
        for i, _ in enumerate(states_):
            if my_state == silence:
                if int(states_[i] % 2) != 0:
                    # Znaleziony onset.
                    last_onset = i * hop_time
                    last_midi = ((states_[i] - 1) / 2) + self.midi_min
                    last_note = librosa.midi_to_note(last_midi)
                    my_state = onset
    
            elif my_state == onset:
                 if int(states_[i] % 2) == 0:
                    # Znaleziona faza brzmienia dźwięku.    
                    my_state = sustain
    
            elif my_state == sustain:
                if int(states_[i] % 2) != 0 or (i in self.__onsets and self.simple_advanced_algorithm):
                    # Znaleziony onset.
                    # Dokończenie poprzedniej nuty.
                    last_offset = i * hop_time
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)
    
                    # Rozpoczęcie nowej nuty.
                    last_onset = i * hop_time
                    last_midi = ((states_[i] - 1) / 2) + self.midi_min if int(states_[i] % 2) != 0 else ((states_[i]) / 2 - 1) + self.midi_min
                    last_note = librosa.midi_to_note(last_midi)
                    my_state = onset
    
                elif states_[i] == 0:
                    # Wykryta cisza. 
                    # Dokończenie poprzedniej nuty.
                    last_offset = i * hop_time
                    my_note = [last_onset, last_offset, last_midi, last_note]
                    output.append(my_note)
                    my_state = silence
        
        for sequence, i in enumerate(output):
            print(sequence, i)
        print(self.__onsets)
        return output
    
    def __sinusoidal_estimation(self, audio_signal: np.array, fs: int, threshold: float = 0.5):
        D = librosa.stft(audio_signal)
        # freq: Instantaneous frequencies, freqs[..., f, t] is the frequency for bin f, frame t
        # reassigned_times: reassigned_times[..., f, t] is the time for bin f, frame t, 
        #mags: Magnitudes from short-time Fourier transform. mags[..., f, t] is the magnitude for bin f, frame t
        freq, reassigned_times, mags = librosa.reassigned_spectrogram(y=audio_signal, sr=fs, S=D)
        Ft, WFt = [], []
        for t, t_unit in enumerate(freq):
            for i, j in combinations(range(1, len(t_unit)), 2):
                    # Get the frequencies and magnitudes of the peaks
                    x, y = t_unit[i], t_unit[j]
                    if(isnan(x) or isnan(y) or x == 0 or y == 0):
                        continue
                    mx, my = mags[t][i], mags[t][j]
        
                    # Ensure y >= x
                    while True:
                        if y < x:
                            x, y = y, x
                            mx, my = my, mx
        
                        # Compute the ratio
                        r = abs((y / x)-round(y/x))
                        if(r < threshold):
                            break
                        else:
                            # Perform the operations until r < 1
                            z = y % x
                            y = z
        
                    # The pitch candidate is gcd(x,y), and the weight is the product of the magnitudes
                    Ft.append((x+y)/(1+round(y/x)))
                    WFt.append(mx * my)
        
        # Output the pitch candidates and their weights
        print("Pitch Candidates:", Ft)
        
        print("Weights:", WFt)
        # plt.figure(figsize=(10, 6))
        # plt.subplot(1, 1, 1)
        # plt.plot(freq)
        # plt.title("Freq")
        # plt.tight_layout()
        # plt.show()
        
        # magnitude, phase = librosa.magphase(D)
        # power_spectrum = np.square(magnitude)
        # peaks, _ = librosa.find_peaks(power_spectrum)
        # print("Peaks: ", peaks)
        # Compute the real and imaginary parts of the STFT
        # a = np.real(D)
        # b = np.imag(D)
        
        # # Compute the derivatives of `a` and `b` with respect to time
        # da_dt = np.gradient(a, axis=-1)
        # db_dt = np.gradient(b, axis=-1)
        
        # # Compute the instantaneous frequency
        # k = (a * db_dt - b * da_dt) / (a**2 + b**2)
        
        # # Compute the amplitude of the IF
        # g = np.abs(D)
        magnitude, phase = librosa.magphase(D)
        # Compute the power spectrum
        power_spectrum = np.abs(magnitude)**2
        

        # Find the peaks in the power spectrum
        peaks = np.array([signal.find_peaks(row)[0] for row in power_spectrum])
        # signal.find_peaks(row)[0] for row in power_spectrum

        #Plot the power_spectrrum and the magnitude on one plot
        plt.figure(figsize=(10, 6))
        plt.subplot(3, 1, 1)
        plt.plot(power_spectrum)
        plt.title("Power Spectrum")
        plt.subplot(3, 1, 2)
        plt.plot(magnitude)
        plt.title("Magnitude")
        plt.subplot(3, 1, 3)
        plt.plot(peaks)
        plt.title("Peaks")
        plt.tight_layout()
        plt.show()
        
        
        # Initialize lists to store the pitch candidates and their weights
        Ft = []
        WFt = []
        
        # Iterate over all pairs of peaks
        for t in peaks:
            for i in range(len(peaks)):
                for j in range(i+1, len(peaks)):
                    # Get the frequencies and magnitudes of the peaks
                    x, y = peaks[i], peaks[j]
                    mx, my = magnitude[i], magnitude[j]
        
                    # Ensure y >= x
                    if y < x:
                        x, y = y, x
                        mx, my = my, mx
        
                    # Compute the ratio
                    r = y / x
        
                    # Perform the operations until r < 1
                    while r >= 1:
                        z = y % x
                        y = z
                        r = y / x
        
                    # The pitch candidate is x, and the weight is the ratio of the magnitudes
                    Ft.append(x)
                    WFt.append(mx / my)
        
        # Output the pitch candidates and their weights
        print("Pitch Candidates:", Ft)
        
        print("Weights:", WFt)
        

    def wave_to_notes_basic(self)-> (list, int):
       # self.__sinusoidal_estimation(audio_signal, srate)
       print("Calculate onsets")
       self.__calculate_onsets_and_f0_candidates(self.__audio_data, self.__sampling_rate, self.frame_length, self.hop_length)
       
       self.__notesroll = self.__onsets_to_notesroll(self.hop_length / self.__sampling_rate)
       self.__bpm = floor(librosa.beat.tempo(y=self.__audio_data)[0])
       return self.__notesroll, self.__bpm


    def wave_to_notes_viterbi(self, p_stay_note: float = 0.7, p_stay_silence: float = 0.7, pitch_acc: float = 0.9, voiced_acc: float = 0.9,
            onset_acc: float = 0.9, ratio: float = 0.9, spread: float = 0.2)-> (list, int):
        # self.__sinusoidal_estimation(audio_signal, srate)
        self.__calculate_onsets_and_f0_candidates(self.__audio_data, self.__sampling_rate, self.frame_length, self.hop_length)
        transmat = self.__calculate_transition_matrix(p_stay_note, p_stay_silence, ratio)
        priors = self.__calculate_prior_probabilities(self.__audio_data, self.__sampling_rate, self.frame_length, self.hop_length, pitch_acc,
            voiced_acc, onset_acc, spread)
        p_init = np.zeros(transmat.shape[0])
        p_init[0] = 1
        states = librosa.sequence.viterbi(priors, transmat, p_init=p_init)
        
       # # Write the 1D array to a text file
       #  np.savetxt('transmat.txt', transmat)
        
       #  # Write the 2D array to a text file
       #  np.savetxt('priors.txt', priors)
       #  np.savetxt('states.txt', states)
        
        
        self.__notesroll = self.__states_to_notesroll(states, self.hop_length / self.__sampling_rate)
        self.__bpm = floor(librosa.beat.tempo(y=self.__audio_data)[0])
        return self.__notesroll, self.__bpm
    

    def __round_to_fraction_power_of_two(self, n, precision=0.2):
        i = 1
        while True:
            power_of_two = 2 ** i
            closest_fraction = round(n * power_of_two) / power_of_two
            if abs(n - closest_fraction) < precision:
                frac = fractions.Fraction(closest_fraction).limit_denominator()
                return frac.numerator, frac.denominator
            i += 1
    
    def __breakdown_fraction(self, numerator, denominator):
        results = []
        while numerator > 0:
            power_of_two = 1
            while power_of_two <= numerator / 2:
                power_of_two *= 2
            results.append((1, denominator // power_of_two))
            numerator -= power_of_two
        return results
    
    def __split_duration(self, duration: abjad.Duration, left_space: abjad.Duration) -> np.array:
        if duration - left_space > 0:
            return np.array([left_space, duration - left_space])
        else:
            return np.array([duration])
                
    def notesroll_to_music_sheet(self, output_directory: str = 'results', error: float = 0.05, precision: float = 0.2, meter=(4, 4)):
        # notes_data = [(60, 'C4', 1000), (62, 'D4', 500), (64, 'E4', 2000), (65, 'F4', 1000)]  # (midi_number, name, duration_ms)
        # s = stream.Stream()
        #     # last_onset, last_offset, last_midi, last_note
           
        # for midi_number, name, duration_ms in notes_data:
        #     n = note.Note(midi=midi_number)
        #     n.duration.quarterLength = duration_ms / 1000.0 * 4  # Convert duration from ms to quarter lengths
        #     s.append(n)
        
        # Show the music sheet
        # s.show()

        container = abjad.Container()
        time_signature = abjad.Meter(meter)
        measures_in_bar = meter[0]
        basic_meter_unit = meter[1]
        basic_meter_unit_time = self.__bpm/60 * (meter[1]/4)

        current_bar = abjad.Duration(meter)
        for note in self.__notesroll:
            onset = float(note[0])
            offset = float(note[1])
            height = note[3].replace("♯", "s")
            relation = (offset - onset) * basic_meter_unit_time
            if abs(relation - np.round(relation)) < error * np.round(relation) and np.round(relation) % 2 != 1:
              note_length = abjad.Duration(1, 4/(2**round(math.log2(relation))))
              durations = self.__split_duration(note_length, current_bar)
              for i, duration in enumerate(durations):
                  current_note = abjad.Note(height, duration)
                  if len(durations) > 1 and i == 0:
                      abjad.attach(abjad.StartSlur(), current_note)
                  elif len(durations) > 1 and i == len(durations) - 1:
                      abjad.attach(abjad.StopSlur(), current_note)
                  container.append(current_note)
                  current_bar -= duration
                  if current_bar == 0:
                      current_bar = abjad.Duration(meter)
    
            else:
              numerator, denominator = self.__round_to_fraction_power_of_two(relation, precision)
              measures = self.__breakdown_fraction(numerator, denominator * 4)
              # if(len(measures) == 1):

              for i, measure in enumerate(measures):
                  durations = self.__split_duration(abjad.Duration(measure), current_bar)
                  for j, duration in enumerate(durations):
                    current_note= abjad.Note(height, duration)
                    if ( (i==0 and len(measures) > 1) or (j==0 and len(durations) > 1)):
                        abjad.attach(abjad.StartSlur(), current_note)
                    elif ((i==len(measures)-1 and len(measures) > 1) or (j==len(durations)-1 and len(durations) > 1)):
                        abjad.attach(abjad.StopSlur(), current_note)
                    current_bar -= duration
                    if current_bar == 0:
                        current_bar = abjad.Duration(meter)
                    container.append(current_note)
                  
        os.environ['PATH'] = str(lilypond.executable().parent) + os.pathsep + os.environ['PATH']
        lilypond_file = abjad.LilyPondFile([container])
        abjad.persist.as_png(lilypond_file, os.path.join(output_directory, 'result.png'))
        abjad.persist.as_pdf(lilypond_file, os.path.join(output_directory, 'result.pdf'))


        # Add the LilyPond directory to the PATH
        # env['lilypondPath'] = path
        # print(env['lilypondPath'])
        # s.write('lily.pdf', fp='output.pdf')
        # lilypond.executable() -fpng -dresolution=300 -dpreview -o preview/ my_file.ly

    def notesroll_to_music_sheet2(self, error: float = 0.05, precision: float = 0.2, time_signature=(4, 4)):
        s = stream.Stream()
        s.append(meter.TimeSignature(f'{time_signature[0]}/{time_signature[1]}'))

        basic_meter_unit_time = 60/self.__bpm * (4/time_signature[1])

        for note_info in self.__notesroll:
            onset, offset, midi_number, name = note_info
            name = name.replace("♯", "#")
            relation = (offset - onset) * basic_meter_unit_time
            if abs(relation - round(relation)) < error * np.round(relation):
                note_length = 1 / (2**round(np.log2(relation)))
            else:
                numerator, denominator = self.__round_to_fraction_power_of_two(relation, precision)
                note_length = numerator / (denominator * 4)
            n = note.Note(name)
            n.duration = duration.Duration(note_length)
            s.append(n)


    # def __round_to_fraction_power_of_two(self, n, precision=0.2):
    #     i = 1
    #     while True:
    #         power_of_two = 2 ** i
    #         closest_fraction = round(n * power_of_two) / power_of_two
    #         if abs(n - closest_fraction) < precision:
    #             frac = fractions.Fraction(closest_fraction).limit_denominator()
    #             return frac.numerator, frac.denominator
    #         i += 1

