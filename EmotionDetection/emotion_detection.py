import requests
import json


def emotion_detector(text_to_analyse):
    url = 'https://sn-watson-emotion.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/EmotionPredict'
    myobj = {"raw_document": {"text": text_to_analyse}}
    headers = {"grpc-metadata-mm-model-id":
               "emotion_aggregated-workflow_lang_en_stock"}
    response = requests.post(url, json=myobj, headers=headers)
    formatted_response = json.loads(response.text)
    if response.status_code == 200:
        return formatted_response
    elif response.status_code == 400:
        formatted_response = {
            'anger': None,
            'disgust': None,
            'fear': None,
            'joy': None,
            'sadness': None,
            'dominant_emotion': None}
        return formatted_response


def emotion_predictor(detected_text):
    if all(value is None for value in detected_text.values()):
        return detected_text
    if detected_text['emotionPredictions'] is not None:
        emotions = detected_text['emotionPredictions'][0]['emotion']
        anger = emotions['anger']
        disgust = emotions['disgust']
        fear = emotions['fear']
        joy = emotions['joy']
        sadness = emotions['sadness']
        max_emotion = max(emotions, key=emotions.get)
        # max_emotion_score = emotions[max_emotion]
        formated_dict_emotions = {
                                'anger': anger,
                                'disgust': disgust,
                                'fear': fear,
                                'joy': joy,
                                'sadness': sadness,
                                'dominant_emotion': max_emotion
                                }
        return formated_dict_emotions


if __name__ == "__main__":
    sample_text = "I'm so happy and excited about the new project!"
    detected_emotions = emotion_detector(sample_text)
    predictions = emotion_predictor(detected_emotions)
    print(predictions)
