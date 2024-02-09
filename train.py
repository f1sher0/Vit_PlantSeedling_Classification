import os

import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import torchvision.datasets as datasets
from models.vision_transformer import deit_tiny_patch16_224, deit_base_patch16_224
from torchtoolbox.tools import mixup_data, mixup_criterion
from torchtoolbox.transform import Cutout

# 设置全局参数
modellr = 1e-4
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# 数据预处理7

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    Cutout(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3281186, 0.28937867, 0.20702125], std=[0.09407319, 0.09732835, 0.106712654])

])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3281186, 0.28937867, 0.20702125], std=[0.09407319, 0.09732835, 0.106712654]),
])
# 读取数据
dataset_train = datasets.ImageFolder('data/train', transform=transform)
dataset_test = datasets.ImageFolder("data/val", transform=transform_test)
print(dataset_train.class_to_idx)
# 导入数据
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# 实例化模型并且移动到GPU
criterion = nn.CrossEntropyLoss()
model_ft = deit_tiny_patch16_224(pretrained=True)
print(model_ft)
num_ftrs = model_ft.head.in_features
model_ft.head = nn.Linear(num_ftrs, 12, bias=True)
nn.init.xavier_uniform_(model_ft.head.weight)
model_ft.to(DEVICE)
print(model_ft)
# 选择简单暴力的Adam优化器，学习率调低
optimizer = optim.Adam(model_ft.parameters(), lr=modellr)
cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=20, eta_min=1e-9)

# 定义训练过程
alpha = 0.2
train_losses = []
val_losses = []
val_accuracies = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    sum_loss = 0
    total_num = len(train_loader.dataset)
    print(total_num, len(train_loader))
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
        data, labels_a, labels_b, lam = mixup_data(data, target, alpha)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = mixup_criterion(criterion, output, labels_a, labels_b, lam)
        loss.backward()
        optimizer.step()
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print_loss = loss.data.item()
        sum_loss += print_loss
        if (batch_idx + 1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR:{:.9f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                       100. * (batch_idx + 1) / len(train_loader), loss.item(), lr))
    ave_loss = sum_loss / len(train_loader)
    print('epoch:{},loss:{}'.format(epoch, ave_loss))
    return ave_loss  # Return the average loss


ACC = 0


# 验证过程
def val(model, device, test_loader, epoch):
    global ACC
    model.eval()
    test_loss = 0
    correct = 0
    total_num = len(test_loader.dataset)
    print(total_num, len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            loss = criterion(output, target)
            _, pred = torch.max(output.data, 1)
            correct += torch.sum(pred == target)
            print_loss = loss.data.item()
            test_loss += print_loss
        correct = correct.data.item()
        acc = correct / total_num
        avgloss = test_loss / len(test_loader)
        print('\nVal set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            avgloss, correct, len(test_loader.dataset), 100 * acc))
        if acc > ACC or (epoch % 5 == 0):
            torch.save(model_ft, 'model_' + str(epoch) + '_' + str(round(acc, 3)) + '.pth')
            ACC = acc
        return avgloss, acc

    # 训练

for epoch in range(1, EPOCHS + 1):
    train_loss = train(model_ft, DEVICE, train_loader, optimizer, epoch)
    train_losses.append(train_loss)  # Save training loss
    cosine_schedule.step()
    val_loss, acc = val(model_ft, DEVICE, test_loader, epoch)
    val_losses.append(val_loss)  # Save validation loss
    val_accuracies.append(acc)  # Save validation accuracy

    # Optionally, write metrics to a file after each epoch or at the end of training
    with open('training_metrics.txt', 'a') as f:
        f.write(f'Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {acc}\n')