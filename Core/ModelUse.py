# ml_model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# 你的模型结构
class SimpleCNNClassifier(nn.Module):
    def __init__(self, num_classes=15, dropout_rate=0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 112x112
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 56x56
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 28x28
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 14x14
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------- 模型加载 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNNClassifier(num_classes=15)
model.load_state_dict(torch.load(os.path.join('Core','Model','CNN.pth'), map_location=device))  # 载入训练好的权重
model.to(device)
model.eval()

# ---------------- 预处理函数 ----------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

CLASS_NAMES = [f"{i}" for i in range(15)]  # TODO: 改成你真实的类别名称
Classnames = {
    0: "彩椒细菌斑点病",
    1: "彩椒健康",
    2: "马铃薯早疫病",
    3: "马铃薯晚疫病",
    4: "马铃薯健康",
    5: "番茄细菌斑点病",
    6: "番茄早疫病",
    7: "番茄晚疫病",
    8: "番茄叶霉病",
    9: "番茄斑点病（Septoria叶斑病）",
    10: "番茄红蜘蛛（双斑螨）",
    11: "番茄靶斑病",
    12: "番茄黄叶卷叶病毒",
    13: "番茄马赛克病毒",
    14: "番茄健康",
}
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return Classnames[predicted.item()]
