import requests

url = "http://your-address/predict"
files = {"image": open("./dataset/input/testImg.jpg", "rb")}

response = requests.post(url, files=files)

print(f"📌 응답 코드: {response.status_code}")  # print HTTP state code

if response.status_code == 200:
    print(response.json())  # normal state.
    
else:
    print(f"❌ 서버 오류: {response.text}")  # print error message