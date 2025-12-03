import torch
import torch.nn as nn


class ShallowCNNforContextVersion1(nn.Module):
    """
    コンテキスト画像用の浅いCNN特徴量抽出器である。
    入力層に大きなカーネルサイズを持つ。フルスクラッチから学習される。
    """
    def __init__(self, in_channels: int = 3, num_features_out: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=16, stride=4, padding=6),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(256, num_features_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features_out),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.num_features = num_features_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        return x
    

class ShallowCNNforContext(nn.Module):
    """
    コンテキスト画像用の浅いCNN特徴量抽出器である。
    入力層に大きなカーネルサイズを持つ。フルスクラッチから学習される。
    """
    def __init__(self, in_channels: int = 3, num_features_out: int = 512):
        super().__init__()
        self.features = nn.Sequential(
            # 第1層: 大きめのカーネルで大域的な特徴を捉える
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第2層: 小さなカーネルで局所的な特徴を抽出
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        
        # AdaptiveAvgPoolで空間次元を(1, 1)に集約
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        
        # 最終的な出力次元を調整するための全結合層
        self.projection = nn.Linear(64, num_features_out)
        self.num_features = num_features_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.projection(x)
        return x