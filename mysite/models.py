from django.db import models

# Create your models here.

class top(models.Model):
    usa = models.IntegerField()
    kor = models.IntegerField()
    chn = models.IntegerField()
    jpn = models.IntegerField()
    deu = models.IntegerField()
    fra = models.IntegerField()
    gbr = models.IntegerField()
    ita = models.IntegerField()
