# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from sklearn.metrics import roc_curve,auc,f1_score, precision_recall_curve, average_precision_score
import shutil
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import glob
import cv2
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示中文标签
log_dir = os.path.join('tensorboard', 'valid', 'vgg')
valid_writer = SummaryWriter(log_dir=log_dir)


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


epochs = 15 # 训练次数
batch_size = 4  # 批处理大小
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
PATH = 'D:/cats_vs_dogs/models/model_vgg.pt'


# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.5),
    transforms.ColorJitter(hue=0.5),
    transforms.ColorJitter(contrast=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = models.vgg16(pretrained=True)
net_new = net
for param in net_new.parameters():
    param.requires_grad = False
net_new.classifier._modules['6'] = nn.Linear(4096, 2)
net_new.classifier._modules['7'] = torch.nn.LogSoftmax(dim=1)
net_new = net_new.to(device)

if os.path.exists('models/model_vgg.pt'):
    net_new = torch.load('models/model_vgg.pt')

if use_gpu:
    net_new = net_new.cuda()
data_input = Variable(torch.randn(16, 3, 32, 32))
print(summary(net_new, (3, 32, 32)))

# 定义loss和optimizer
cirterion = nn.NLLLoss()
optimizer_vgg = torch.optim.Adam(net_new.classifier[6].parameters(), lr=0.0005)


# 在某个数据集上检测正确率，测试模型
def valid_model(model, dataloader, size):
    # model.eval(). 否则的话，有输入数据，即使不训练，它也会改变权值。
    # eval()就是保证BN和dropout不发生变化，框架会自动把BN和DropOut固定住，不会取平均，
    # 而是用训练好的值，不然的话，一旦test的batch_size过小，很容易就会被BN层影响结果！！！
    model.eval()
    # .zeros 初始化为0
    predictions = np.zeros(size)
    all_classes = np.zeros(size)
    all_proba = np.zeros((size,2))
    i = 0
    running_loss = 0.0
    running_corrects = 0
    # 对于dataloader 这个加载器中的每一个变量
    for inputs, classes in dataloader:
        # 将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        if use_gpu:
            inputs, classes = Variable(inputs.cuda()), Variable(classes.cuda())
        else:
            inputs, classes = Variable(inputs), Variable(classes)
        outputs = model(inputs)
        prob = F.softmax(outputs, dim=1)
        loss = cirterion(outputs, classes)
        # torch.max()简单来说是返回一个tensor中的最大值。
        _, preds = torch.max(outputs.data,1)
        # statistics
        running_loss += loss.data.item()
        running_corrects += torch.sum(preds == classes.data)
        # t.numpy()将Tensor变量转换为ndarray变量，其中t是一个Tensor变量，
        # 可以是标量，也可以是向量，转换后dtype与Tensor的dtype一致。
        predictions[i:i+len(classes)] = preds.to('cpu').numpy()
        all_classes[i:i+len(classes)] = classes.to('cpu').numpy()
        all_proba[i:i+len(classes),:] = prob[0][0].item() if prob[0][0].item()>prob[0][1].item() else prob[0][1].item()
        i += len(classes)
        # print('Testing: No. ', i, ' process ... total: ', size)
    epoch_loss = running_loss / size
    epoch_acc = running_corrects.data.item() / size
    print('Loss: {:.4f} Acc: {:.4f}'.format(
                     epoch_loss, epoch_acc))
    return predictions, all_proba, all_classes


def train():
    Loss_list_train = []
    Accuracy_list_train = []
    Loss_list_valid = []
    Accuracy_list_valid = []
    x1 = range(0+1, epochs+1)
    x2 = range(0+1, epochs+1)
    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for i, data in enumerate(train_loader, 0):
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            outputs = net_new(inputs)

            optimizer_vgg.zero_grad()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer_vgg.step()
            _, train_predicted = torch.max(outputs.data, 1)
            # print(train_predicted)
            train_correct += (train_predicted == labels.data).sum()
            running_loss += loss.item()
            train_total += labels.size(0)
        Loss_list_train.append(running_loss / train_total)
        Accuracy_list_train.append(float(100 * train_correct / train_total))
        print('train %d epoch loss: %.3f  acc: %.3f ' % (
                epoch + 1, running_loss / train_total, 100 * train_correct / train_total))

        # 模型验证
        correct = 0
        valid_loss = 0.0
        valid_total = 0
        net_new.eval()
        for data in valid_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net_new(images)
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted)
            loss = cirterion(outputs, labels)
            valid_loss += loss.item()
            valid_total += labels.size(0)
            correct += (predicted == labels.data).sum()
        Loss_list_valid.append(valid_loss / valid_total)
        Accuracy_list_valid.append(float(100 * correct / valid_total))
        valid_writer.add_scalar('Accuracy', float(100 * correct / valid_total), epoch + 1)
        valid_writer.add_scalar('Loss', valid_loss / valid_total, epoch + 1)

        print('valid %d epoch loss: %.3f  acc: %.3f '
              % (epoch + 1, valid_loss / valid_total, 100 * correct / valid_total))
    torch.save(net_new, 'models/model_vgg.pt')

    # y1_train = Accuracy_list_train
    # y2_train = Loss_list_train
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1_train, 'o-')
    # plt.title('Train accuracy vs. epoches')
    # plt.ylabel('Train accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2_train, '.-')
    # plt.xlabel('Train loss vs. epoches')
    # plt.ylabel('Train loss')
    # plt.show()
    # plt.savefig("accuracy_loss.jpg")
    #
    # y1_valid = Accuracy_list_valid
    # y2_valid = Loss_list_valid
    # plt.subplot(2, 1, 1)
    # plt.plot(x1, y1_valid, 'o-')
    # plt.title('Valid accuracy vs. epoches')
    # plt.ylabel('Valid accuracy')
    # plt.subplot(2, 1, 2)
    # plt.plot(x2, y2_valid, '.-')
    # plt.xlabel('Valid loss vs. epoches')
    # plt.ylabel('Valid loss')
    # plt.show()
    # plt.savefig("accuracy_loss.jpg")


def test():
    probs_cat = []
    probs_dog = []
    classes = ('cat', 'dog')
    model = torch.load('models/model_vgg.pt')
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
        if pred_class == 'cat':
            probs_cat.append(float(prob[0][0].item()*100 if prob[0][0].item() > prob[0][1].item() else prob[0][1].item()*100))
        else:
            probs_dog.append(float(prob[0][0].item()*100 if prob[0][0].item() > prob[0][1].item() else prob[0][1].item()*100))

        img = cv2.imread(paths[i])
        cv2.imshow(paths[i].removeprefix(s) + ' in ' + st + ' prob is a ' + pred_class, img)
        cv2.waitKey(2000)
        cv2.destroyWindow(paths[i].removeprefix(s) + ' in ' + st + ' prob is a ' + pred_class)

    # plt.boxplot(probs_cat)
    # plt.show()


def valid():
    model = torch.load('models/model_vgg.pt')
    model.eval()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    valid_set = datasets.ImageFolder(root='D:/cats_vs_dogs/data/valid/', transform=valid_transforms)
    valid_loader = DataLoader(valid_set, batch_size=1, drop_last=False, shuffle=False)
    valid_predictions, valid_all_proba, valid_all_classes = valid_model(model,
                                                                        valid_loader,
                                                                        size=len(valid_loader.dataset))
    # with open("cat_dog_results/cat_dog_vgg_result.csv", 'w') as out:
    #     for i in range(len(valid_loader.dataset)):
    #         out.write("{} {} {}\n".format(valid_all_classes[i], valid_predictions[i], valid_all_proba[i][0]))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = []
    y_score = []
    with open("cat_dog_results/cat_dog_vgg_result.csv", 'r') as fs:
        for line in fs:
            nonclk, clk, score = line.strip().split(' ')
            nonclk = 0 if nonclk=='0.0' else 1
            clk = 0 if clk=='0.0' else 1
            y_test.append(nonclk)
            y_score.append(clk)
    fpr, tpr, threshold = roc_curve(y_test, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.figure(figsize=(10,10))
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve(area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy',lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
# 绘制ROC曲线


# init()
# train()
test()
# valid()
