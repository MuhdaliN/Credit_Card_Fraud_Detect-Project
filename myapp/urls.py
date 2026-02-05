# myproject/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('home/', views.home, name='home'),
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('upload-csv/', views.upload_csv_view, name='upload_csv'),
    path('delete-csv/', views.delete_csv_view, name='delete_csv'),
    path('analyze-csv/', views.analyze_csv_view, name='analyze_csv'),  # ADD THIS
    path('advanced-analysis/', views.advanced_analysis_view, name='advanced_analysis'),
    path('download-report/', views.download_analysis_report, name='download_analysis_report'),
    path('get-analysis-data/', views.get_analysis_data, name='get_analysis_data'),
    path('faq/', views.faq_view, name='faq'),
    path('debug-ml/', views.debug_ml_status, name='debug_ml'),
    path('logout/', views.logout_view, name='logout'),
]