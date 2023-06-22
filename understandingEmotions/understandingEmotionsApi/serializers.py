from rest_framework import serializers

class StringMessageSerializer(serializers.Serializer):
    message = serializers.CharField(required=True)

class EmotionSerializer(serializers.Serializer):
    rating = serializers.FloatField(required=True)
    emotion = serializers.CharField(required=True)