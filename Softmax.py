from torch import nn
from uitls.softmax_utils import *


class Softmax(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(num_input, num_output)
        self.init_parameters()

    def forward(self, input_data):
        return self.linear(self.flatten(input_data))

    def init_parameters(self):
        for net in self.modules():
            if isinstance(net, nn.Linear):
                nn.init.normal_(net.weight, 0, 0.01)
                nn.init.constant_(net.bias, 0)


# 读取Fashion-MNIST数据集
batch_size = 256
train_iter, test_iter = load_fashion_mnist_data(batch_size)

# 定义模型
model = Softmax(28*28, 10)
loss = nn.CrossEntropyLoss(reduction='none')
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

# 训练
epochs = 10
train_softmax(model, train_iter, test_iter, loss, epochs, optimizer)

# 输出预测结果
predict_softmax(model, test_iter)