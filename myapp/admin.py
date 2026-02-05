from django.contrib import admin
from .models import User_db, UserCSVFile

# Register your models here.
admin.site.register(User_db)
admin.site.register(UserCSVFile)
