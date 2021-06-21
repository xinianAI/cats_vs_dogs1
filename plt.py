import pandas as pd
import matplotlib.pyplot as plt


def accuracy():
    plt.title('Accuracy of Nets of CNN in Validation datasets')
    net1 = pd.read_csv('run-valid_alexnet-tag-Accuracy.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net1.Value, lw=1.5, label='alexnet', color='pink')
    net2 = pd.read_csv('run-valid_vgg-tag-Accuracy.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net2.Value, lw=1.5, label='vgg', color='green')
    net3 = pd.read_csv('run-valid_googlenet-tag-Accuracy.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net3.Value, lw=1.5, label='googlenet', color='red')
    net4 = pd.read_csv('run-valid_resnet-tag-Accuracy.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net4.Value, lw=1.5, label='resnet', color='purple')
    net5 = pd.read_csv('run-valid_densenet-tag-Accuracy.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net5.Value, lw=1.5, label='densenet', color='black')

    plt.legend(loc="lower right")
    plt.show()


def loss():
    plt.title('Loss of Nets of CNN in Validation datasets')
    net1 = pd.read_csv('run-valid_alexnet-tag-Loss.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net1.Value, lw=1.5, label='alexnet', color='pink')
    net2 = pd.read_csv('run-valid_vgg-tag-Loss.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net2.Value, lw=1.5, label='vgg', color='green')
    net3 = pd.read_csv('run-valid_googlenet-tag-Loss.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net3.Value, lw=1.5, label='googlenet', color='red')
    net4 = pd.read_csv('run-valid_resnet-tag-Loss.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net4.Value, lw=1.5, label='resnet', color='purple')
    net5 = pd.read_csv('run-valid_densenet-tag-Loss.csv', usecols=['Step', 'Value'])
    plt.plot(net1.Step, net5.Value, lw=1.5, label='densenet', color='black')

    plt.legend(loc="lower right")
    plt.show()


accuracy()
loss()
