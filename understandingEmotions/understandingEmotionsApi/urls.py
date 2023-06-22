from django.urls import path, include
from rest_framework.routers import DefaultRouter

from .views import understandingEmotionsView

router = DefaultRouter()
router.register(r'understandingEmotions', understandingEmotionsView, basename='understandingEmotions')

urlpatterns = [
    path('', include(router.urls))
]