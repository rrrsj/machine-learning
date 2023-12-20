import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils import data
import torchvision
import torch
from uitls.Animator import Animator


def get_fashion_mnist_labels(labels):
    """
    返回Fashion-MNIST数据集的文本标签。
    例如标签 index[1, 4] 返回 ['trouser', 'coat']
    :param labels: List of int | index
    :return: List of str
    """
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[label] for label in labels]


def set_load_threads_num():
    """设置加载数据集的线程数"""
    return 4


def load_fashion_mnist_data(batch_size, resize=None):
    """
    从互联网下载Fashion-MNIST数据集，存储在 "./data/Fashion-MNIST" 文件夹内。
    并加载数据集到data.dataloader中。
    :param batch_size: int
    :param resize: tuple ()
    :return: train_iter, test_iter
    """
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (
        data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True,
                        num_workers=set_load_threads_num()),
        data.DataLoader(mnist_test, batch_size=batch_size, shuffle=True,
                        num_workers=set_load_threads_num()))


def accuracy(y_pre, y):
    """计算预测正确的样本个数"""
    if len(y_pre) > 1 and y_pre.shape[1] > 1:
        y_pre = y_pre.argmax(axis=1)
    cmp = (y == y_pre.type(y.dtype))
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:
    """累加器"""
    def __init__(self,  n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [item + float(para) for item, para in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def train_epoch_softmax(model, train_iter, lossFunc, optim):
    """训练模型的一个迭代周期"""
    if isinstance(model,  torch.nn.Module):
        model.train()
    info = Accumulator(3)
    for x, y in train_iter:
        y_hat = model.forward(x)
        loss = lossFunc(y_hat, y)
        if isinstance(optim, torch.optim.Optimizer):
            optim.zero_grad()
            loss.mean().backward()
            optim.step()
        info.add(float(loss.sum()), accuracy(y_hat, y), y.numel())
    return info[0] / info[2], info[1] / info[2]


def train_softmax(model, train_iter, test_iter, lossFunc, num_epochs, optim):
    """训练模型，并画图"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_info = train_epoch_softmax(model, train_iter, lossFunc, optim)
        test_acc = evaluate_accuracy(model, test_iter)
        animator.add(epoch + 1, train_info + (test_acc, ))
    train_loss, train_acc = train_info
    plt.show()
    assert train_loss < 0.5, train_loss
    assert 1 >= train_acc > 0.7, train_acc
    assert 1 >= test_acc > 0.7, test_acc


def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images.

    Defined in :numref:`sec_utils`"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        try:
            img = d2l.numpy(img)
        except:
            pass
        ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes


def predict_softmax(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = get_fashion_mnist_labels(y)
    preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(
        X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()
