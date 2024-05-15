import requests

url = 'http://localhost:5000/predict'
api_key = 'passone'
headers = {'API-Key': api_key}
data = {'text': 'Congratulations! Youve won a free vacation to an exotic island. Click here to claim your prize now!'}

response = requests.post(url, json=data, headers=headers)

print(response.json())
