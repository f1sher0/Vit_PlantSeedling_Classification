import torch.utils.data.distributed
import torchvision.transforms as transforms
from PIL import Image
from torch.autograd import Variable
import os

classes = ('Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed',
           'Common wheat', 'Fat Hen', 'Loose Silky-bent',
           'Maize', 'Scentless Mayweed', 'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet')
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3281186, 0.28937867, 0.20702125], std=[0.09407319, 0.09732835, 0.106712654]),
])
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load("model_3_0.921.pth")
model.eval()
model.to(DEVICE)

path = 'data/test/'
testList = os.listdir(path)
for file in testList:
    img = Image.open(path + file)
    img = transform_test(img)
    img.unsqueeze_(0)
    img = img.to(DEVICE)
    out, attentions = model(img)

    # Predict
    _, pred = torch.max(out.data, 1)
    print('Image Name:{},predict:{}'.format(file, classes[pred.data.item()]))