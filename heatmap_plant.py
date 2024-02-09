import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from transformers import ViTFeatureExtractor, ViTForImageClassification
import matplotlib.pyplot as plt
import numpy as np


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model_41_0.96.pth")
model.eval()
model.to(DEVICE)


image = Image.open("./data/train/Charlock/0e51b1876.png")
transform = Compose([Resize((224, 224)),
                     ToTensor(),
                     Normalize(mean=[0.3281186, 0.28937867, 0.20702125],
                               std=[0.09407319, 0.09732835, 0.106712654])])

img = transform(image).unsqueeze(0)
img = img.to(DEVICE)
# 获取attention权重
_, attentions = model(img)  # 获取所有层的attention权重 (12, 1, 3, 197, 197)


# 计算所有头的平均attention
avg_attention = torch.mean(torch.stack(attentions), dim=2)  # 在头维度上平均

for i in range(len(avg_attention)):
    attention = avg_attention[i]  # shape (1, 197, 197)

    attention = attention[0, 0, 1:].reshape(14, 14).cpu().detach().numpy()  # 移除CLS token，调整形状 shape (14, 14)


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
