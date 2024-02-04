import io
import os
import zipfile
from .apps import RecordingToSheetMusicConverterConfig
from django.shortcuts import render
from django.http import JsonResponse, FileResponse
from django.conf import settings
from rest_framework.views import APIView
from pydub import AudioSegment
import tempfile

            
# Create your views here.
class audio_converter(APIView):
    model = RecordingToSheetMusicConverterConfig.model
    def post(self, request):
        if request.method == 'POST':
            audio_file = request.FILES['file']  # get the uploaded file
            # file_name = default_storage.save('audios/' + audio_file.name, audio_file)  # save the file
            # handle_uploaded_file(request.FILES["file"])
            # file = io.BytesIO(audio_file.read())
            extension = os.path.splitext(audio_file.name)[1].lower()
            self.model = RecordingToSheetMusicConverterConfig.model

            if extension == '.wav':
                # For WAV files, we can directly load them with Librosa
                tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                for chunk in audio_file.chunks():
                    tmp.write(chunk)
                tmp.close()
                self.model.initialize_audio_data(tmp.name)
                os.unlink(tmp.name)
            elif extension in ['.mp3', '.flac']:
                # For MP3 and FLAC files, we need to convert them to WAV first
                tmp = tempfile.NamedTemporaryFile(suffix=extension[1:], delete=False)
                for chunk in audio_file.chunks():
                    tmp.write(chunk)
                tmp.close()
                self.model.initialize_audio_data(tmp.name)
                os.unlink(tmp.name)
            RecordingToSheetMusicConverterConfig.model = self.model
            return JsonResponse({'message': 'Audio file uploaded successfully', 'file_name': audio_file.name})

        return JsonResponse({'message': 'Invalid request method'}, status=400)
    def get(self, request):
        if request.method == "GET":
            params = {}
            output_dir = 'results'
            for key, value in request.GET.items():
                params[key] = value
                print(key, value)
            # text = request.GET.get("text")

            self.model.initialize_audio_parameters(params.get('start_note'), params.get('end_note'), params.get('onset_method'))
            conversion_method = params.get('conversion_method')
            if(int(conversion_method) == 0):
                print('Wav to notes basic')
                self.model.wave_to_notes_basic()
            else:
                print('Wav to notes viterbi')
                self.model.wave_to_notes_viterbi(p_stay_note = float(params.get('sustain_probability')), p_stay_silence = float(params.get('silence_probability')), pitch_acc = float(params.get('f0_probability')),
                                                 onset_acc = float(params.get('onset_probability')), ratio = float(params.get('ratio')))                     
            self.model.notesroll_to_music_sheet(output_directory=output_dir)
                
            # Utworzenie pliku zip 
            temp = tempfile.TemporaryFile(suffix='.zip', delete = False)
            # zip_file_path = os.path.join(temp, 'results.zip')
            with zipfile.ZipFile(temp, 'w') as zip_file:
                # Add files to the zip file
                files = os.listdir(output_dir)
                for file_name in files:
                    file_path = os.path.join(output_dir, file_name)
                    zip_file.write(file_path, arcname=file_name)
            
            # Send the zip file in the response
            response = FileResponse(open(temp.name, 'rb'))
            response['Content-Disposition'] = f'attachment; filename={temp}'
            temp.close()
            return response
            # response = {'text_sentiment': text}
            # return JsonResponse(response)