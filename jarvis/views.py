from django.shortcuts import render
from django.http import JsonResponse
from rest_framework.decorators import api_view
from .NLP_MK1 import chatbot


# Create your views here.

@api_view(['GET', 'POST'])
def initialse(request):
    global jarvis
    jarvis = chatbot('testbot.h5', 0.98)
    jarvis.print_parameters()
    return JsonResponse(request.data, safe=False)


@api_view(['GET', 'POST'])
def request_query(request):
    print(request.data['ask'])
    res, intent = jarvis.ask(request.data['ask'])

    print("------" ,res , intent)
    resp = {

        "response": res,
        "intent": intent

    }

    return JsonResponse(resp, safe=False)
