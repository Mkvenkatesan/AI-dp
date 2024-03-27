# image_prediction_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_image, name='predict_image'),
]
