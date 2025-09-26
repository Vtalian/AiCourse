from django.db import models

# Create your models here.

class Disease(models.Model):
    name = models.CharField(max_length=100, unique=True,primary_key=True)
    harm=models.TextField(max_length=500)
    description = models.TextField(max_length=500)
    precaution = models.TextField(max_length=500)
    solution = models.TextField(max_length=500)

    def __str__(self):
        return self.name