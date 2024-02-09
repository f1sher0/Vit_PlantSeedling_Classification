import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTFeatureExtractor, ViTForImageClassification
import matplotlib.pyplot as plt
import numpy as np

# 加载预训练的ViT模型和特征提取器
model_name = 'google/vit-base-patch16-224-in21k'
model = ViTForImageClassification.from_pretrained(model_name)
feature_extractor = ViTFeatureExtractor.from_pretrained(model_name)

# 准备图像
image = Image.open("./data/train/Charlock/0e51b1876.png")
transform = Compose([Resize((224, 224)), ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
input_tensor = transform(image).unsqueeze(0)

# 获取attention权重
outputs = model(input_tensor, output_attentions=True)
attentions = outputs.attentions  # 获取所有层的attention权重 (12, 1, 12, 197, 197)

# 计算所有头的平均attention
avg_attention = torch.mean(torch.stack(attentions), dim=2)  # 在头维度上平均

for i in range(len(avg_attention)):
    attention = avg_attention[i]  # shape (1, 197, 197)

    attention = attention[0, 0, 1:].reshape(14, 14).detach().numpy()  # 移除CLS token，调整形状 shape (14, 14)

    # 重新调整attention heatmap的大小以匹配原始图像
    attention_resized = np.array(Image.fromarray(attention).resize(image.size, Image.BILINEAR))
    # 可视化heatmap
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(image)
    plt.imshow(attention_resized, cmap='jet', alpha=0.4)  # alpha调节透明度
    plt.title("Attention Heatmap for Layer {layer}".format(layer=i))
    plt.show()
