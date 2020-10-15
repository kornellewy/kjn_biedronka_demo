from django import forms
import os
from django.conf import settings

class UploadFieldForm(forms.Form):
    movie_name = forms.CharField(max_length=1000)
    movie_field = forms.FileField(widget=forms.ClearableFileInput(attrs={'multiple': False}))