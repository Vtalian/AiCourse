from django.urls import path
from . import views

app_name = "detectApp"
urlpatterns = [
    path("", views.index, name="index"),
    path("detectimg/", views.detectimg, name="detectimg"),
]
