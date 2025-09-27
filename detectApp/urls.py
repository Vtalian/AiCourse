from django.urls import path
from . import views

app_name = "detectApp"
urlpatterns = [
    path("", views.index, name="index"),
    path("detect", views.detect, name="detect"),
    path("detectimg/", views.detectimg, name="detectimg"),
    path("diseases", views.diseases, name="diseases"),
    path("disease/<str:name>/", views.disease_detail, name="disease_detail"),
]
