from django.shortcuts import render
import joblib
import json, os
import numpy as np
import pandas as pd
import psycopg2
from .models import regression

# Create your views here.
def index(request):
    return render(request, "index.html")

def result(request):
    model = joblib.load('../models/models.joblib')
    list = []

    list.append(float(request.GET['age']))
    list.append(float(request.GET['sex']))
    list.append(float(request.GET['bmi']))
    list.append(float(request.GET['children']))
    list.append(float(request.GET['smoker']))
    list.append(float(request.GET['region']))

    answer = model.predict([list])[0]

    b = regression(
        age=request.GET['age'], 
        sex=request.GET['sex'], 
        bmi=request.GET['bmi'], 
        children=request.GET['children'], 
        smoker=request.GET['smoker'], 
        region=request.GET['region'], 
        charges=answer
    )

    b.save()
    return render(request, "index.html", {'answer':answer})





