import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes = [5,9,17]):
        super(InceptionModule, self).__init__()
        BT_channels = in_channels // 4 if (in_channels // 4) > 0 else in_channels
        self.bottleneck = nn.Conv1d(in_channels, BT_channels, kernel_size=1, stride=1, padding=0) if (in_channels // 4) > 0 else nn.Identity()
        self.conv1 = nn.Conv1d(BT_channels, out_channels//4, kernel_size=kernel_sizes[0], stride=1, padding=(kernel_sizes[0]-1)//2)
        self.conv2 = nn.Conv1d(BT_channels, out_channels//4, kernel_size=kernel_sizes[1], stride=1, padding=(kernel_sizes[1]-1)//2)
        self.conv3 = nn.Conv1d(BT_channels, out_channels//4, kernel_size=kernel_sizes[2], stride=1, padding=(kernel_sizes[2]-1)//2)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.conv_maxpool = nn.Conv1d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        BT = self.bottleneck(x)
        x1 = self.conv1(BT)
        x2 = self.conv2(BT)
        x3 = self.conv3(BT)
        x4 = self.conv_maxpool(self.maxpool(x))
        out = torch.cat([x1, x2, x3, x4], dim=1)
        return F.relu(self.batch_norm(out))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[5, 9, 17]):
        super(ResidualBlock, self).__init__()
        self.inception1 = InceptionModule(in_channels, out_channels, kernel_sizes=kernel_sizes)
        self.inception2 = InceptionModule(out_channels, out_channels, kernel_sizes=kernel_sizes)
        self.inception3 = InceptionModule(out_channels, out_channels, kernel_sizes=kernel_sizes)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        return F.relu(x + shortcut)#원소별 덧셈

class InceptionNetwork(nn.Module):
    def __init__(self, input_channels, num_classes, kernel_sizes=[5,9,17], residual_channels=[64,128,128], fc_channels=[64,32], dropout=0):
        super(InceptionNetwork, self).__init__()
        self.residual_block1 = ResidualBlock(input_channels, residual_channels[0], kernel_sizes = kernel_sizes)
        self.residual_block2 = ResidualBlock(residual_channels[0], residual_channels[1], kernel_sizes = kernel_sizes)
        self.residual_block3 = ResidualBlock(residual_channels[1], residual_channels[2], kernel_sizes = kernel_sizes)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)#원하는 차원으로 풀링되는게 맞나?
        self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(residual_channels[1], fc_channels[0])
        # self.fc2 = nn.Linear(fc_channels[0], fc_channels[1])
        # self.fc3 = nn.Linear(fc_channels[1],num_classes)
        
        # FC layer 구성 (유연하게)
        layers = []
        in_dim = residual_channels[-1]  # residual block 출력 차원
        
        for out_dim in fc_channels:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        
        # 마지막 출력 레이어
        layers.append(nn.Linear(in_dim, num_classes))  # 회귀용이면 num_outputs=1

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.residual_block1(x)
        x = self.residual_block2(x)
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        
        # #print(x)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        #x = F.relu(x)
        x = self.fc_layers(x)
        return x

# model = InceptionNetwork(3,23)
# print(model)