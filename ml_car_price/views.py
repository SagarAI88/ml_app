from django.shortcuts import render
from django.http import HttpResponse
import pickle
import numpy as np

# Create your views here.

def home(request):
    return render(request, 'home.html')

def pred(request):
    
    f1 = float(request.GET['f1'])
    f2 = int(request.GET['f2'])
    f3 = int(request.GET['f3'])
    f4 = int(request.GET['f4'])
    f5 = int(request.GET['f5'])
    f6 = int(request.GET['f6'])
    f7 = int(request.GET['f7'])
    f8 = int(request.GET['f8'])
    
    loaded_model = pickle.load(open('/home/sagar/ml_app/ml_car_price/finalized_model.sav', 'rb'))

    result = loaded_model.predict(np.array([[f1,f2,f3,f4,f5,f6,f7,f8]]))
    
    return render(request,'result.html',{'result':result})
