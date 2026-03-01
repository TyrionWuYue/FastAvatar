import torch
import torch.nn as nn
import torch.nn.functional as F
import math

__all__ = ['ArcFaceLoss']

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class IRBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, use_se=True):
        super(IRBlock, self).__init__()
        self.bn0 = nn.BatchNorm2d(inplanes)
        self.conv1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.prelu = nn.PReLU()
        self.conv2 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(planes)
    def forward(self, x):
        residual = x
        out = self.bn0(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.use_se:
            out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.prelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, use_se=True):
        self.inplanes = 64
        self.use_se = use_se
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(512 * 7 * 7, 512)
        self.bn3 = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_se=self.use_se))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_se=self.use_se))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.bn3(x)
        return x


class ArcMarginModel(nn.Module):
    def __init__(self, args):
        super(ArcMarginModel, self).__init__()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, args.emb_size))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = args.easy_margin
        self.m = args.margin_m
        self.s = args.margin_s

        self.cos_m = math.cos(self.m)
        self.sin_m = math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    def forward(self, input, label):
        x = F.normalize(input)
        W = F.normalize(self.weight)
        cosine = F.linear(x, W)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m  # cos(theta + m)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class ArcFaceLoss(nn.Module):
    def __init__(self, device, arcface_checkpoint="./model_zoo/arcface/arcface_checkpoint.tar"):
        super(ArcFaceLoss, self).__init__()
        self.device = device
        
        import sys
        import types
        models_module = types.ModuleType('models')
        arcface_models = types.ModuleType('arcface_models')
        models_module.arcface_models = arcface_models
        sys.modules['models'] = models_module
        sys.modules['models.arcface_models'] = arcface_models
        
        arcface_models.ResNet = ResNet
        arcface_models.IRBlock = IRBlock
        arcface_models.SEBlock = SEBlock
        arcface_models.ArcMarginModel = ArcMarginModel

        
        checkpoint = torch.load(arcface_checkpoint, map_location=torch.device("cpu"), weights_only=False)
        self.netArc = checkpoint.to(device)
        self.netArc.eval()
    
    def cosin_metric(self, x1, x2):
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))
    
    def forward(self, pred, target):
        # process multi-frame input
        if pred.dim() == 5:  # [batch, frames, channels, height, width]
            batch_size, num_frames, channels, height, width = pred.shape
            
            # [batch*frames, channels, height, width]
            pred_reshaped = pred.view(batch_size * num_frames, channels, height, width)
            target_reshaped = target.view(batch_size * num_frames, channels, height, width)
        elif pred.dim() == 4:  # [batch, channels, height, width]
            pred_reshaped = pred
            target_reshaped = target
        else:
            raise ValueError(f"Expected 4D or 5D input, got {pred.dim()}D")
        
        pred_resized = F.interpolate(pred_reshaped, size=(112, 112), mode='bilinear', align_corners=False)
        target_resized = F.interpolate(target_reshaped, size=(112, 112), mode='bilinear', align_corners=False)
        
        pred_normalized = (pred_resized - 0.5) * 2.0
        target_normalized = (target_resized - 0.5) * 2.0
        
        latent_pred = self.netArc(pred_normalized)
        latent_pred = F.normalize(latent_pred, p=2, dim=1)
        latent_target = self.netArc(target_normalized)
        latent_target = F.normalize(latent_target, p=2, dim=1)

        loss = (1 - self.cosin_metric(latent_pred, latent_target)).mean()
        
        return loss