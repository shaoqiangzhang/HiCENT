import torch
import torch.nn as nn
from torchvision.models import resnet50

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=4):
        super(ResNetFeatureExtractor, self).__init__()
        # 加载预训练的ResNet50模型
        model = resnet50(pretrained=True)
        # 提取前几层作为特征提取网络
        self.features = nn.Sequential(*list(model.children())[:feature_layer + 1])
        # 冻结所有参数
        for param in self.features.parameters():
            param.requires_grad = False

    def forward(self, x):
        # 确保输入有三个通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.features(x)

class FeatureConsistencyLoss(nn.Module):
    def __init__(self):
        super(FeatureConsistencyLoss, self).__init__()
        self.resnet_extractor = ResNetFeatureExtractor(feature_layer=4)  # 可以根据需要调整层级
        self.mse_loss = nn.MSELoss()

    def forward(self, output_images, target_images):
        output_features = self.resnet_extractor(output_images)
        target_features = self.resnet_extractor(target_images)
        return self.mse_loss(output_features, target_features)

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.tv_loss = TVLoss()
        self.feature_consistency_loss = FeatureConsistencyLoss()
    def forward(self, out_images, target_images):
        feature_consistency_loss = self.feature_consistency_loss(out_images, target_images)
        # Image Loss
        image_loss = self.mse_loss(out_images, target_images)
        tv_loss = self.tv_loss(out_images)
        # L1 Loss
        l1_loss = self.l1_loss(out_images, target_images)
        # KL Divergence Loss
        total_loss = image_loss + 0.01 * feature_consistency_loss + 1e-5 * tv_loss + l1_loss
        return total_loss

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        b, c, h, w = x.shape
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w-1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / b

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]