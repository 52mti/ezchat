import requests


response = requests.post('http://localhost:9999/tts' , json= {'text': "测试音频播放"})
print(response.json())