from django.apps import AppConfig
from django.conf import settings
from django.db.models.base import Model
from api.RecordingToSheetMusicConverter import models
import os


class RecordingToSheetMusicConverterConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "api.RecordingToSheetMusicConverter"
    path = os.path.join(settings.BASE_DIR, "RecordingToSheetMusicConverter")
    model = models.Model()

