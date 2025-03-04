{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 7,
      "metadata": {
        "id": "ufdRf-E-0IDj"
      },
      "outputs": [],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torchvision\n",
        "import torchvision.transforms as transforms\n",
        "from torch.utils.data import DataLoader"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ugHagGz0wAJ7"
      },
      "source": [
        "<h1>Data transform\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "0WuGaERJ0NAT"
      },
      "outputs": [],
      "source": [
        "transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),  # VGG-16 입력 크기\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화\n",
        "])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "Hp_0fqUPwuY0"
      },
      "outputs": [],
      "source": [
        "# 데이터셋 로드 (예제에서는 ImageFolder 사용)\n",
        "train_dataset = torchvision.datasets.ImageFolder(root='./dataset/training_set/training_set', transform=transform)\n",
        "test_dataset = torchvision.datasets.ImageFolder(root='./dataset/test_set/test_set', transform=transform)\n",
        "\n",
        "train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n",
        "test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "-3IZbWYRwvXE"
      },
      "outputs": [],
      "source": [
        "import torch.nn as nn\n",
        "\n",
        "class VGG16(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(VGG16, self).__init__()\n",
        "\n",
        "        self.conv_layers = nn.Sequential(\n",
        "            # Conv Block 1\n",
        "            nn.Conv2d(3, 64, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(64, 64, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(kernel_size=2, stride=2),\n",
        "\n",
        "            # Conv Block 2\n",
        "            nn.Conv2d(64, 128, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(128, 128, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(kernel_size=2, stride=2),\n",
        "\n",
        "            # Conv Block 3\n",
        "            nn.Conv2d(128, 256, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(256, 256, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(256, 256, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(kernel_size=2, stride=2),\n",
        "\n",
        "            # Conv Block 4\n",
        "            nn.Conv2d(256, 512, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(512, 512, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(512, 512, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(kernel_size=2, stride=2),\n",
        "\n",
        "            # Conv Block 5\n",
        "            nn.Conv2d(512, 512, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(512, 512, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.Conv2d(512, 512, kernel_size=3, padding=1),\n",
        "            nn.ReLU(),\n",
        "            nn.MaxPool2d(kernel_size=2, stride=2)\n",
        "        )\n",
        "\n",
        "        self.fc_layers = nn.Sequential(\n",
        "            nn.Flatten(),\n",
        "            nn.Linear(512 * 7 * 7, 4096),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.5),\n",
        "            nn.Linear(4096, 4096),\n",
        "            nn.ReLU(),\n",
        "            nn.Dropout(0.5),\n",
        "            nn.Linear(4096, 1),  # 🔹 2개 클래스 → 1개 출력 노드\n",
        "            nn.Sigmoid()  # 🔹 이진 분류를 위한 Sigmoid 추가\n",
        "        )\n",
        "\n",
        "    def forward(self, x):\n",
        "        x = self.conv_layers(x)\n",
        "        x = self.fc_layers(x)\n",
        "        return x  # Sigmoid 값 출력 (0~1 사이 확률값)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 25,
      "metadata": {
        "id": "yssYhts6ySRn"
      },
      "outputs": [
        {
          "name": "stderr",
          "output_type": "stream",
          "text": [
            "c:\\Users\\tnqja\\anaconda3\\envs\\AI\\lib\\site-packages\\torchvision\\models\\_utils.py:208: UserWarning: The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future, please use 'weights' instead.\n",
            "  warnings.warn(\n",
            "c:\\Users\\tnqja\\anaconda3\\envs\\AI\\lib\\site-packages\\torchvision\\models\\_utils.py:223: UserWarning: Arguments other than a weight enum or `None` for 'weights' are deprecated since 0.13 and may be removed in the future. The current behavior is equivalent to passing `weights=VGG16_Weights.IMAGENET1K_V1`. You can also use `weights=VGG16_Weights.DEFAULT` to get the most up-to-date weights.\n",
            "  warnings.warn(msg)\n",
            "Downloading: \"https://download.pytorch.org/models/vgg16-397923af.pth\" to C:\\Users\\tnqja/.cache\\torch\\hub\\checkpoints\\vgg16-397923af.pth\n",
            "100%|██████████| 528M/528M [00:09<00:00, 57.0MB/s] \n"
          ]
        },
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch [1/20], Loss: 0.0619\n",
            "Epoch [2/20], Loss: 0.0126\n",
            "Epoch [3/20], Loss: 0.0077\n",
            "Epoch [4/20], Loss: 0.0069\n",
            "Epoch [5/20], Loss: 0.0108\n",
            "Epoch [6/20], Loss: 0.0106\n",
            "Epoch [7/20], Loss: 0.0246\n",
            "Epoch [8/20], Loss: 0.0175\n",
            "Epoch [9/20], Loss: 0.0073\n",
            "Epoch [10/20], Loss: 0.0064\n",
            "Epoch [11/20], Loss: 0.0048\n",
            "Epoch [12/20], Loss: 0.0025\n",
            "Epoch [13/20], Loss: 0.0056\n",
            "Epoch [14/20], Loss: 0.0125\n",
            "Epoch [15/20], Loss: 0.0066\n",
            "Epoch [16/20], Loss: 0.0162\n",
            "Epoch [17/20], Loss: 0.0137\n",
            "Epoch [18/20], Loss: 0.0058\n",
            "Epoch [19/20], Loss: 0.0018\n",
            "Epoch [20/20], Loss: 0.0116\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.optim as optim\n",
        "import torchvision.models as models\n",
        "\n",
        "# ✅ VGG-16 모델 수정 (출력층: 1개 노드, Sigmoid 필요)\n",
        "class VGG16Binary(nn.Module):\n",
        "    def __init__(self):\n",
        "        super(VGG16Binary, self).__init__()\n",
        "        self.vgg16 = models.vgg16(pretrained=True)  # Pretrained VGG-16 불러오기\n",
        "\n",
        "        # 🔹 Feature Extractor (Conv Layers) 고정 (Fine-tuning)\n",
        "        for param in self.vgg16.features.parameters():\n",
        "            param.requires_grad = False\n",
        "\n",
        "        # 🔹 Fully Connected Layer 수정 (출력층 1개 → Sigmoid 사용)\n",
        "        self.vgg16.classifier[6] = nn.Linear(4096, 1)\n",
        "\n",
        "    def forward(self, x):\n",
        "        return self.vgg16(x)  # Sigmoid 없이 logits 출력\n",
        "\n",
        "# ✅ 모델 생성\n",
        "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n",
        "model = VGG16Binary().to(device)\n",
        "\n",
        "# ✅ 손실 함수 변경 (CrossEntropyLoss → BCEWithLogitsLoss)\n",
        "criterion = nn.BCEWithLogitsLoss()\n",
        "optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 학습률 낮춤\n",
        "\n",
        "# ✅ 학습 과정\n",
        "num_epochs = 20\n",
        "for epoch in range(num_epochs):\n",
        "    model.train()\n",
        "    total_loss = 0\n",
        "\n",
        "    for images, labels in train_loader:\n",
        "        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # 🔹 Float 변환 및 차원 변경\n",
        "\n",
        "        optimizer.zero_grad()\n",
        "        outputs = model(images)  # Logits 값 출력\n",
        "        loss = criterion(outputs, labels)  # 손실 계산\n",
        "        loss.backward()\n",
        "        optimizer.step()\n",
        "\n",
        "        total_loss += loss.item()\n",
        "\n",
        "    print(f\"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 30,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Validation Loss: 0.5051\n",
            "Validation Accuracy: 97.92%\n",
            "F1 Score: 0.9796\n"
          ]
        }
      ],
      "source": [
        "import torch\n",
        "import torch.nn as nn\n",
        "from torchmetrics.classification import BinaryF1Score\n",
        "\n",
        "# ✅ 모델을 평가 모드로 변경\n",
        "model.eval()\n",
        "\n",
        "# ✅ 검증(Validation) 데이터 평가\n",
        "correct = 0\n",
        "total = 0\n",
        "total_loss = 0\n",
        "\n",
        "# ✅ Precision, Recall, F1 Score 계산을 위한 리스트\n",
        "all_labels = []\n",
        "all_preds = []\n",
        "\n",
        "# 손실 함수 정의 (BCEWithLogitsLoss 사용)\n",
        "criterion = nn.BCEWithLogitsLoss()\n",
        "\n",
        "with torch.no_grad():\n",
        "    for images, labels in test_loader:\n",
        "        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)  # 🔹 Float 변환\n",
        "\n",
        "        outputs = model(images)\n",
        "        loss = criterion(outputs, labels)  # 🔹 손실 계산\n",
        "        total_loss += loss.item()\n",
        "\n",
        "        preds = torch.sigmoid(outputs) > 0.5  # 🔹 0.5 이상이면 1(강아지), 아니면 0(고양이)\n",
        "\n",
        "        correct += (preds == labels).sum().item()\n",
        "        total += labels.size(0)\n",
        "\n",
        "        # 리스트에 추가\n",
        "        all_labels.append(labels)\n",
        "        all_preds.append(preds)\n",
        "\n",
        "# ✅ 정확도(Accuracy) 계산\n",
        "accuracy = 100 * correct / total\n",
        "avg_loss = total_loss / len(test_loader)\n",
        "\n",
        "print(f\"Validation Loss: {avg_loss:.4f}\")\n",
        "print(f\"Validation Accuracy: {accuracy:.2f}%\")\n",
        "\n",
        "# ✅ F1 Score 계산 (PyTorch Metrics 사용)\n",
        "f1_metric = BinaryF1Score().to(device)\n",
        "f1 = f1_metric(torch.cat(all_preds), torch.cat(all_labels))\n",
        "print(f\"F1 Score: {f1:.4f}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": [
        "#모델 테스트\n",
        "import torch\n",
        "import torchvision.transforms as transforms\n",
        "from PIL import Image\n",
        "\n",
        "# ✅ 이미지 전처리 (VGG-16에 맞게 변환)\n",
        "transform = transforms.Compose([\n",
        "    transforms.Resize((224, 224)),  # VGG-16 입력 크기\n",
        "    transforms.ToTensor(),\n",
        "    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet 정규화\n",
        "])\n",
        "\n",
        "def preprocess_image(image_path):\n",
        "    \"\"\"이미지를 로드하고 VGG-16 입력 형태로 변환\"\"\"\n",
        "    image = Image.open(image_path).convert(\"RGB\")  # 이미지를 RGB 모드로 변환\n",
        "    image = transform(image)  # 변환 적용\n",
        "    image = image.unsqueeze(0)  # 배치 차원 추가 (모델은 [batch, C, H, W] 형태 입력 필요)\n",
        "    return image.to(device)\n",
        "\n",
        "import torch.nn.functional as F\n",
        "\n",
        "def predict_image(model, image_path):\n",
        "    \"\"\"이미지를 입력받아 모델이 강아지/고양이 예측\"\"\"\n",
        "    model.eval()  # 모델을 평가 모드로 설정\n",
        "    image = preprocess_image(image_path)  # 이미지 전처리\n",
        "    \n",
        "    with torch.no_grad():\n",
        "        output = model(image)\n",
        "        probability = torch.sigmoid(output).item()  # 확률값 변환 (Sigmoid 적용)\n",
        "        prediction = \"Dog 🐶\" if probability > 0.5 else \"Cat 🐱\"\n",
        "    \n",
        "    print(f\"Image: {image_path} | Predicted: {prediction} | Probability: {probability:.4f}\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {},
      "outputs": [
        {
          "ename": "NameError",
          "evalue": "name 'model' is not defined",
          "output_type": "error",
          "traceback": [
            "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "Cell \u001b[1;32mIn[6], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m predict_image(\u001b[43mmodel\u001b[49m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m./dataset/input/dog1.jpg\u001b[39m\u001b[38;5;124m\"\u001b[39m) \n",
            "\u001b[1;31mNameError\u001b[0m: name 'model' is not defined"
          ]
        }
      ],
      "source": [
        "predict_image(model, \"./dataset/input/dog1.jpg\") "
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [
        {
          "ename": "NameError",
          "evalue": "name 'torch' is not defined",
          "output_type": "error",
          "traceback": [
            "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
            "Cell \u001b[1;32mIn[3], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# ✅ 모델 학습 후 저장 (예제에서는 학습이 완료된 상태라고 가정)\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m \u001b[43mtorch\u001b[49m\u001b[38;5;241m.\u001b[39msave({\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmodel_state_dict\u001b[39m\u001b[38;5;124m\"\u001b[39m: model\u001b[38;5;241m.\u001b[39mstate_dict()}, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmodel.pth\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28mprint\u001b[39m(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m✅ 모델이 model.pth 파일로 저장되었습니다.\u001b[39m\u001b[38;5;124m\"\u001b[39m)\n",
            "\u001b[1;31mNameError\u001b[0m: name 'torch' is not defined"
          ]
        }
      ],
      "source": [
        "\n",
        "# ✅ 모델 학습 후 저장 (예제에서는 학습이 완료된 상태라고 가정)\n",
        "torch.save(model.state_dict(), \"model.pth\")  # 🔹 학습된 가중치만 저장\n",
        "\n",
        "print(\"✅ 모델이 model.pth 파일로 저장되었습니다.\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 34,
      "metadata": {},
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Requirement already satisfied: flask in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (3.0.3)\n",
            "Requirement already satisfied: Werkzeug>=3.0.0 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from flask) (3.0.4)\n",
            "Requirement already satisfied: Jinja2>=3.1.2 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from flask) (3.1.4)\n",
            "Requirement already satisfied: itsdangerous>=2.1.2 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from flask) (2.2.0)\n",
            "Requirement already satisfied: click>=8.1.3 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from flask) (8.1.7)\n",
            "Requirement already satisfied: blinker>=1.6.2 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from flask) (1.8.2)\n",
            "Requirement already satisfied: importlib-metadata>=3.6.0 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from flask) (8.0.0)\n",
            "Requirement already satisfied: colorama in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from click>=8.1.3->flask) (0.4.6)\n",
            "Requirement already satisfied: zipp>=0.5 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from importlib-metadata>=3.6.0->flask) (3.19.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\tnqja\\anaconda3\\envs\\ai\\lib\\site-packages (from Jinja2>=3.1.2->flask) (2.1.5)\n",
            "Note: you may need to restart the kernel to use updated packages.\n"
          ]
        }
      ],
      "source": [
        "pip install flask"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "AI",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.9.19"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
