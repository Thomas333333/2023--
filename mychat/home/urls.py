from django.urls import path,re_path
from . import views

urlpatterns = [
	path('index/',views.index),
	path('getanswer/',views.getanswer),
	path('runoob/',views.runoob),
    path('getAnswer/', views.getAnswer, name='getAnswer'),
    path('clear-history/', views.clearHistory, name='clearHistory'),
]