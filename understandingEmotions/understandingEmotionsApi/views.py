import os
import json
import pickle

from django.http import JsonResponse
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.viewsets import ViewSet

from .predict import predict_emotion_for
from .serializers import *
from .startSession import start_session
from .classification import classify

def load_vectorizer(vectorizer_dir, vectorizer_name):
    with open(os.path.join(vectorizer_dir, vectorizer_name), 'rb') as f:
        vectorizer, probs = pickle.load(f)
    return vectorizer, probs

model_dir = "onnx"
model_name = "model.onnx"
vectorizer_name = ["vectorizer1.pkl"]

session, tokenizer = start_session(model_dir, model_name)
vectorizer, probs = load_vectorizer("vectorizers", vectorizer_name[0])

class understandingEmotionsView(ViewSet):
    @swagger_auto_schema(
        operation_id="predictEmotion",
        operation_description="predicting emotions by text",
        operation_summary="Return emotion statistics",
        request_body=StringMessageSerializer(),
        responses={
            200: EmotionSerializer(),
            400: StringMessageSerializer()
        }
    )
    @action(methods=['post'], detail=False, url_path="predictEmotion")
    def predict_emotion(self, request: Request):
        """ Builds three routes using time, length and weight optimizations and returns all of them"""

        decoded_body = request.body.decode('utf-8')
        body = json.loads(decoded_body)

        validation = StringMessageSerializer(data=body)

        if validation.is_valid():
            message = body["message"]

            emotion_rating, emotion = predict_emotion_for(message, session, tokenizer)
            response = {
                "Score": emotion_rating,
                "Emotion": emotion
            }

            return JsonResponse(response, status=status.HTTP_200_OK)

        return JsonResponse({"message": "Validation exception!"}, status=status.HTTP_400_BAD_REQUEST)
    
    @swagger_auto_schema(
        operation_id="eventClassification",
        operation_description="classification of events by description",
        operation_summary="Return type of event",
        request_body=StringMessageSerializer(),
        responses={
            200: StringMessageSerializer(),
            400: StringMessageSerializer()
        }
    )
    @action(methods=['post'], detail=False, url_path="eventClassification")
    def event_classification(self, request: Request):
        """ Builds three routes using time, length and weight optimizations and returns all of them"""

        decoded_body = request.body.decode('utf-8')
        body = json.loads(decoded_body)

        validation = StringMessageSerializer(data=body)

        if validation.is_valid():
            description = body["message"]

            label = classify(description, vectorizer, probs)
            response = {
                "label": label
            }

            return JsonResponse(response, status=status.HTTP_200_OK)

        return JsonResponse({"message": "Validation exception!"}, status=status.HTTP_400_BAD_REQUEST)
