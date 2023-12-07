import torch

def train_model(data_iter, epochs, model, optimizer, lossFunction):
    history_loss = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (x, y) in data_iter:
            optimizer.zero_grad()
            output = model.forward(x)
            loss = lossFunction(output, y.view(-1, 1))
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        history_loss.append(epoch_loss)
        # if epoch % 10 == 1:
        print("epoch{} loss: {}".format(epoch, epoch_loss))
    return history_loss


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