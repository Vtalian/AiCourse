# ml_model.py
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os


# 你的模型结构
class CNNClassifierConvNeXt(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.2, model_type="convnext_tiny"):
        super().__init__()

        # 加载预训练的ConvNeXt模型
        if model_type == "convnext_tiny":
            self.backbone = models.convnext_tiny(pretrained=True)
        elif model_type == "convnext_small":
            self.backbone = models.convnext_small(pretrained=True)
        elif model_type == "convnext_base":
            self.backbone = models.convnext_base(pretrained=True)
        else:
            self.backbone = models.convnext_tiny(pretrained=True)

        # 正确获取特征维度
        # 对于ConvNeXt模型，需要先获取分类器的输入特征数
        if hasattr(self.backbone, "classifier"):
            if isinstance(self.backbone.classifier, nn.Sequential):
                # 查找最后一个线性层
                for layer in reversed(self.backbone.classifier):
                    if isinstance(layer, nn.Linear):
                        in_features = layer.in_features
                        break
                else:
                    # 如果没有找到线性层，使用默认值
                    in_features = 768  # ConvNeXt Tiny的默认特征维度
            else:
                in_features = self.backbone.classifier.in_features
        else:
            # 对于不同的ConvNeXt版本，尝试获取特征维度
            in_features = 768  # 默认值，适用于ConvNeXt Tiny

        print(f"Detected in_features: {in_features}")

        # 替换分类器 - 修复版本
        self.backbone.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 添加全局平均池化
            nn.Flatten(1),  # 展平特征
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------- 模型加载 ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifierConvNeXt()
model.load_state_dict(
    torch.load(os.path.join("Core", "Model", "ConvNeXt.pth"), map_location=device)
)  # 载入训练好的权重
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

CLASS_NAMES = [f"{i}" for i in range(10)]  # TODO: 改成你真实的类别名称
Classnames = {
    0: "番茄细菌斑点病",
    1: "番茄早疫病",
    2: "番茄晚疫病",
    3: "番茄叶霉病",
    4: "番茄斑点病（Septoria叶斑病）",
    5: "番茄红蜘蛛（双斑螨）",
    6: "番茄靶斑病",
    7: "番茄黄叶卷叶病毒",
    8: "番茄花叶病毒",
    9: "番茄健康",
}
def predict_image(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)  # (1,3,224,224)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
    return Classnames[predicted.item()]
