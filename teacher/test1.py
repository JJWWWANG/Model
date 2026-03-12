import torch
import torch.nn as nn

# 假设 batch_size 是 1，你可以根据需要调整这个值
batch_size = 1

# 创建一个形状为 (batch_size, 12, 128) 的随机张量作为输入
x = torch.randn(64,1536)
class VideoModel(nn.Module):
    def __init__(self, num_patches=16):
        super(VideoModel, self).__init__()
        self.fc = nn.Linear(in_features=128 * 12, out_features=7)

    def forward(self, x):
        print("x1.shape",x.size())  #(64, 1536)
        x = self.fc(x)
        print("x2.shape", x.size()) #(64, 1536)
        return x

model = VideoModel()
y = model(x)

# 打印输出张量的形状
print(y.shape)  # 输出应该是 torch.Size([1, 7])

# x1.shape torch.Size([64, 1536])
# x2.shape torch.Size([64, 7])
# torch.Size([64, 7])