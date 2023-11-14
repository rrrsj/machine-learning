import torch
import matplotlib.pyplot as plt
import torch.utils.data as Data
from torch import nn


def generate_linear_data(weight, bias, num_example):
    """

    :param weight: torch.tensor | ndarray | py.List  生成线性数据的权重
    :param bias: float | scaler 偏移量
    :param num_example: int 生成的数据数量
    :return: inputs, outputs
    """
    inputs = torch.normal(0, 1, (num_example, len(weight)))
    outputs = torch.matmul(inputs, weight) + bias
    outputs += torch.normal(0, 0.01, outputs.shape)
    return inputs, outputs


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
optimizer = torch.optim.SGD(LR.parameters(), lr=0.03)

# 训练
history_loss = []
epochs = 5
for epoch in range(epochs):
    epoch_loss = 0.0
    for (x, y) in data_iter:
        optimizer.zero_grad()
        output = LR.forward(x)
        loss = lossFunction(output, y.view(-1, 1))
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    history_loss.append(epoch_loss)
    print("epoch{} loss: {}".format(epoch + 1, epoch_loss))

for name, para in LR.named_parameters():
    print("para: {}    value: {}".format(name, para))

plt.plot(range(1, epochs + 1), history_loss)
plt.xlabel("trained_epoch")
plt.ylabel("Loss of Model")
plt.title("Linear Regression")
plt.show()
