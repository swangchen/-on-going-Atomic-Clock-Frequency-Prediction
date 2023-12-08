import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# 读取数据集
data = np.loadtxt('1.txt', delimiter=',')

# 提取特征和目标
features = data[:, 1:]
target = data[:, 1]

# 数据标准化
scaler = MinMaxScaler(feature_range=(-1, 1))
features_scaled = scaler.fit_transform(features)

# 划分训练集和测试集
train_size = int(len(features) * 0.8)
test_size = len(features) - train_size
train_features, test_features = features_scaled[0:train_size, :], features_scaled[train_size:len(features), :]
train_target, test_target = target[0:train_size], target[train_size:len(target)]

# 将数据转换为PyTorch张量，并将其移至GPU
train_features = torch.Tensor(train_features).to(device)
test_features = torch.Tensor(test_features).to(device)
train_target = torch.Tensor(train_target).to(device)
test_target = torch.Tensor(test_target).to(device)


# 定义LSTM模型，并将其移至GPU
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(device),
                            torch.zeros(1, 1, self.hidden_layer_size).to(device))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


# 实例化模型，并将其移至GPU
model = LSTM(input_size=features.shape[1], hidden_layer_size=100, output_size=1).to(device)

# 定义损失函数和优化器
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

# 训练模型，并收集损失值
epochs = 1000
losses = []

for i in range(epochs):
    optimizer.zero_grad()
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                         torch.zeros(1, 1, model.hidden_layer_size).to(device))

    y_pred = model(train_features)

    loss = loss_function(y_pred, train_target)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % 10 == 0:
        print(f'Epoch {i} Loss: {loss.item()}')

# 绘制损失函数图
plt.plot(losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.show()
