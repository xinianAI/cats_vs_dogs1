# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import glob
import cv2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示中文标签


def init():

    original_dataset_dir = 'D:/cats_vs_dogs/dataset'
    total_num = int(len(os.listdir(original_dataset_dir)) / 2)
    random_idx = np.array(range(total_num))
    np.random.shuffle(random_idx)

    # 待处理的数据集地址
    base_dir = 'D:/cats_vs_dogs/data'
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

    # 训练集、验证集、测试集的划分
    sub_dirs = ['train', 'valid']
    animals = ['cats', 'dogs']
    train_idx = random_idx[:int(total_num * 0.75)]
    valid_idx = random_idx[int(total_num * 0.75):]
    numbers = [train_idx, valid_idx]
    for idx, sub_dir in enumerate(sub_dirs):
        dir = os.path.join(base_dir, sub_dir)
        if not os.path.exists(dir):
            os.mkdir(dir)
        for animal in animals:
            animal_dir = os.path.join(dir, animal)  #
            if not os.path.exists(animal_dir):
                os.mkdir(animal_dir)
            fnames = [animal[:-1] + '.{}.jpg'.format(i) for i in numbers[idx]]
            for fname in fnames:
                src = os.path.join(original_dataset_dir, fname)
                dst = os.path.join(animal_dir, fname)
                shutil.copyfile(src, dst)

            # 验证训练集、验证集、测试集的划分的照片数目
            print(animal_dir + ' total images : %d' % (len(os.listdir(animal_dir))))

    base_dir_test = 'D:/cats_vs_dogs/data'
    sub_dir_test = 'test'
    original_dataset_dir_test = 'D:/cats_vs_dogs/test'
    total_num_test = int(len(os.listdir(original_dataset_dir_test)))
    random_idx_test = np.array(range(1, total_num_test+1))
    np.random.shuffle(random_idx_test)
    if not os.path.exists(base_dir_test):
        os.mkdir(base_dir_test)
    test_idx = random_idx_test[int(total_num_test * 0.75):]
    dir_test = os.path.join(base_dir_test, sub_dir_test)
    if not os.path.exists(dir_test):
        os.mkdir(dir_test)
    fnames_test = ['{}.jpg'.format(i) for i in test_idx]
    for fname_test in fnames_test:
        src = os.path.join(original_dataset_dir_test, fname_test)
        dst = os.path.join(dir_test, fname_test)
        shutil.copyfile(src, dst)


epochs = 50  # 训练次数
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
PATH = 'D:/cats_vs_dogs/models/model_lstm.pt'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 224

# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(5),
    # transforms.ColorJitter(brightness=0.5),
    # transforms.ColorJitter(hue=0.5),
    # transforms.ColorJitter(contrast=0.5),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


train_dataset = datasets.ImageFolder(root='D:/cats_vs_dogs/data/train/',
                                     transform=data_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

valid_dataset = datasets.ImageFolder(root='D:/cats_vs_dogs/data/valid/',
                                     transform=data_transform)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers)

N_STEPS = 224
N_INPUTS = 224
N_NEURONS = 128
N_OUTPUT = 2
N_LAYERS = 1


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.layer_dim = layer_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).requires_grad_().to(device)

        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


net_new = LSTMModel(batch_size,  N_NEURONS, N_LAYERS, N_OUTPUT).to(device)
print(net_new)

if os.path.exists('models/model_lstm.pt'):
    net_new = torch.load('models/model_lstm.pt')
if use_gpu:
    net_new = net_new.cuda()

# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer_lstm = torch.optim.SGD(net_new.parameters(), lr=0.01)

for i in range(len(list(net_new.parameters()))):
    print(list(net_new.parameters())[i].size())


def train():
    Loss_list_valid = []
    Accuracy_list_valid = []
    x1 = range(0+1, epochs+1)
    x2 = range(0+1, epochs+1)
    net_new.train()
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader):
            optimizer_lstm.zero_grad()
            inputs, labels = data
            inputs = inputs.view(-1, 224, 224).requires_grad_().to(device)
            labels = labels.to(device)
            outputs = net_new(inputs)
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer_lstm.step()

            _, train_predicted = torch.max(outputs.data, 1)
            # print(train_predicted)
            train_correct += (train_predicted == labels.data).sum()
            running_loss += loss.item()
            train_total += labels.size(0)
        print('train %d epoch loss: %.3f  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))

        # 模型验证
        correct = 0
        valid_loss = 0.0
        valid_total = 0
        # net_new.eval()
        for data in valid_loader:
            inputs, labels = data
            inputs = inputs.view(-1, 224, 224).requires_grad_().to(device)
            labels = labels.to(device)
            outputs = net_new(inputs)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            loss = cirterion(outputs, labels)
            valid_loss += loss.item()
            valid_total += labels.size(0)
            correct += (predicted == labels.data).sum()
        Loss_list_valid.append(valid_loss / valid_total)
        Accuracy_list_valid.append(float(100 * correct / valid_total))
        print('valid %d epoch loss: %.3f  acc: %.3f '
              % (epoch + 1, valid_loss / valid_total, 100 * correct / valid_total))
    torch.save(net_new, 'models/model_lstm.pt')

    y1 = Accuracy_list_valid
    y2 = Loss_list_valid
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Valid accuracy vs. epoches')
    plt.ylabel('Valid accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Valid loss vs. epoches')
    plt.ylabel('Valid loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")


def test():
    classes = ('cat', 'dog')
    model = torch.load('models/model_lstm.pt')
    model.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    test_set = datasets.ImageFolder(root='D:/cats_vs_dogs/data/test/', transform=test_transforms)
    test_loader = DataLoader(test_set, batch_size=1, drop_last=False, shuffle=False)
    PATH_test = 'D:/cats_vs_dogs/data/test/test'
    s = PATH_test+'\\'
    paths = glob.glob(os.path.join(PATH_test, '*.jpg'))

    for i, data in enumerate(test_loader, 0):
        inputs, train_labels = data
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(train_labels)

        outputs = net_new(inputs)
        prob = F.softmax(outputs, dim=1)
        value, predicted = torch.max(outputs.data, 1)
        pred_class = classes[predicted.item()]
        st = str('%.2f' % float(prob[0][0].item()*100 if prob[0][0].item() > prob[0][1].item() else prob[0][1].item()*100)
                 + '%')
        img = cv2.imread(paths[i])
        cv2.imshow(paths[i].removeprefix(s) + ' in ' + st + ' prob is a ' + pred_class, img)
        cv2.waitKey(2000)
        cv2.destroyWindow(paths[i].removeprefix(s) + ' in ' + st + ' prob is a ' + pred_class)


# init()
# train()
test()
