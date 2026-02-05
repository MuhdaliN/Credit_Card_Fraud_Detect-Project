# models.py
from django.db import models
from django.contrib.auth.models import User
import json

class User_db(models.Model):
    name_db = models.CharField(max_length=100)
    email_db = models.EmailField(unique=True)
    pass_db = models.CharField(max_length=255)

    def __str__(self):
        return self.email_db

class UserCSVFile(models.Model):
    user = models.OneToOneField(User_db, on_delete=models.CASCADE)
    file = models.FileField(upload_to='user_csv_files/')
    original_filename = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    analysis_result = models.JSONField(null=True, blank=True)  # Add this field
    
    def __str__(self):
        return f"{self.user.email_db} - {self.original_filename}"