import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch import nn
from uitls.linear_utils import *


class LinearRegress(nn.Module):
    def __init__(self, num_inputs):
        super().__init__()
        self.linear = nn.Linear(num_inputs, 1)
        self.init_parameter()

    def forward(self, input_data):
        res = self.linear(input_data)
        return res

    def init_parameter(self):
        for net in self.modules():
            if isinstance(net, nn.Linear):
                nn.init.normal_(net.weight, 0, 0.010)
                nn.init.constant_(net.bias, 0)


# 生成数据
true_weight = torch.tensor([11.4, -5.14])
true_bias = 1.19
num_examples = 1000
features, labels = generate_linear_data(true_weight, true_bias, num_examples)

# 数据迭代器
batch_size = 5
dataset = Data.TensorDataset(features, labels)
data_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 模型
LR = LinearRegress(2)
lossFunction = nn.MSELoss()
optim = torch.optim.SGD(LR.parameters(), lr=0.03)

# 训练
epochs = 5
history_loss = train_model(data_iter, model=LR, lossFunction=lossFunction, optimizer=optim, epochs=epochs)

plt.plot(range(1, epochs + 1), history_loss)
plt.xlabel("trained_epoch")
plt.ylabel("Loss of Model")
plt.title("Linear Regression")
plt.show()
