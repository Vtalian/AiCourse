from django.db import models

# Create your models here.

class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True,primary_key=True)
    image = models.ImageField(upload_to="tomatoDiseaseImage/",default="tomatoDiseaseImage/default.jpg")
    harmlevel=models.CharField(max_length=50)
    category=models.CharField(max_length=50)
    harm=models.TextField(max_length=500)
    description = models.TextField(max_length=500)
    precaution = models.TextField(max_length=500)
    solution = models.TextField(max_length=500)

    def __str__(self):
        return self.name
