import json

from django.http import JsonResponse
from drf_yasg.utils import swagger_auto_schema
from rest_framework import status
from rest_framework.decorators import action
from rest_framework.request import Request
from rest_framework.viewsets import ViewSet

from .predict import predict_emotion_for
from .serializers import *
from .startSession import start_session

model_dir = "onnx"
model_name = "model.onnx"

session, tokenizer = start_session(model_dir, model_name)


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
