import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 读取2.txt文件，指定没有header，手动设置列名
df = pd.read_csv('2.txt', header=None, names=['时间', '状态'])

# 数据预处理
scaler = MinMaxScaler()
df[['时间', '状态']] = scaler.fit_transform(df[['时间', '状态']])

# 将数据划分为输入和输出
X = df['时间'].values.reshape(-1, 1)
y = df['状态'].values

# 将数据转换为PyTorch张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# 创建数据集和数据加载器
dataset = TensorDataset(X_tensor, y_tensor)

# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 初始化模型、损失函数和优化器
input_size = 1  # 输入特征的数量
hidden_size = 64  # LSTM隐藏层的大小
num_classes = len(df['状态'].unique())  # 分类的数量
model = LSTMModel(input_size, hidden_size, num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型并记录损失
num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # 训练集上的训练
    model.train()
    for inputs, labels in train_dataloader:
        optimizer.zero_grad()
        outputs = model(inputs.unsqueeze(2))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    train_losses.append(loss.item())

    # 测试集上的测试
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for inputs, labels in test_dataloader:
            outputs = model(inputs.unsqueeze(2))
            test_loss += criterion(outputs, labels).item()
        test_loss /= len(test_dataloader)
        test_losses.append(test_loss)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item()}, Test Loss: {test_loss}')

# 画出损失函数的图像
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

print("训练完成")
