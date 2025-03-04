import torch
import torchvision.transforms as transforms
import torchvision.models as models
from flask import Flask, request, jsonify
from PIL import Image
import os
from flask_cors import CORS
from torchvision.models import resnet50


# ✅ 모델 로드 (Resnet50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ✅ VGG-16 모델 불러오기
model = models.resnet50(pretrained=True)
model.classifier[6] = torch.nn.Linear(4096, 1)  # 이진 분류용 출력층
model.to(device)

# ✅ 저장된 모델 가중치 불러오기 (strict=False 추가)
checkpoint = torch.load("ResNet_CatandDog_classification.pth", map_location=device)
model.load_state_dict(checkpoint, strict=False)
model.eval()

print("✅ Model successfully loaded and ready for inference!")

# ✅ 이미지 전처리 함수
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # VGG-16 입력 크기
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ✅ Flask 앱 생성
app = Flask(__name__)
CORS(app)  # 🔹 CORS 허용
# ✅ 이미지 업로드 디렉토리 설정
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file uploaded"}), 400
        
        file = request.files["image"]
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)  # 이미지 저장

        # 이미지 로드 및 변환
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        # 모델 예측
        with torch.no_grad():
            output = model(image)
            probability = torch.sigmoid(output).item()
            prediction = "Dog 🐶" if probability > 0.5 else "Cat 🐱"

        return jsonify({"prediction": prediction, "probability": probability})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555, debug=True)
