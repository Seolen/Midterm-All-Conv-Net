import torch
import torch.nn as nn
import torch.nn.functional as F

import time

class BaseNet(nn.Module):
    def __init__(self, num_classes=10, **kwargs):
        super(BaseNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 96, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, padding=1, stride=2)

        self.conv3 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv4 = nn.Conv2d(192, 192, 1)
        self.conv5 = nn.Conv2d(192, num_classes, 1)
        # self.global_avg = F.adaptive_avg_pool2d()

    # backhead for A,B,C Net
    def bottom_forward(self, c2):
        p2 = self.pool2(c2)
        c3 = F.relu(self.conv3(p2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))
        out = F.adaptive_avg_pool2d(c5, 1)
        out = (out.squeeze_(-1)).squeeze_(-1)
        return out

    # backhead for SCC,CCC,ACC Net
    def variation_forward(self, p2):
        c3 = F.relu(self.conv3(p2))
        c4 = F.relu(self.conv4(c3))
        c5 = F.relu(self.conv5(c4))
        out = F.adaptive_avg_pool2d(c5, 1)
        out = (out.squeeze_(-1)).squeeze_(-1)
        return out

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        torch.save(self.state_dict(), name)
        return name

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)


class ANet(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        super(ANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, padding=1, kernel_size=5)
        self.conv2 = nn.Conv2d(96, 192, padding=1, kernel_size=5)

    def forward(self, x):
        '''
        x: 32*32 image
        '''
        c1 = F.relu(self.conv1(x))
        p1 = self.pool1(c1)
        c2 = F.relu(self.conv2(p1))

        return self.bottom_forward(c2)


class BNet(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        super(BNet, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 96, padding=1,  kernel_size=5)
        self.conv1_2 = nn.Conv2d(96, 96, padding=1,  kernel_size=1)
        self.conv2 = nn.Conv2d(96, 192, padding=1,  kernel_size=5)
        self.conv2_2 = nn.Conv2d(192, 192, padding=1, kernel_size=1)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c1_2 = F.relu(self.conv1_2(c1))
        p1 = self.pool1(c1_2)
        c2 = F.relu(self.conv2(p1))
        c2_2 = F.relu(self.conv2_2(c2))

        return self.bottom_forward(c2_2)


class CNet(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        super(CNet, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 96, padding=1, kernel_size=3)
        self.conv1_2 = nn.Conv2d(96, 96, padding=1, kernel_size=3)
        self.conv2 = nn.Conv2d(96, 192, padding=1, kernel_size=3)
        self.conv2_2 = nn.Conv2d(192, 192, padding=1, kernel_size=3)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c1_2 = F.relu(self.conv1_2(c1))
        p1 = self.pool1(c1_2)
        c2 = F.relu(self.conv2(p1))
        c2_2 = F.relu(self.conv2_2(c2))

        return self.bottom_forward(c2_2)


class SCCNet(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        super(SCCNet, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 96, padding=1, kernel_size=3)
        self.conv1_2 = nn.Conv2d(96, 96, padding=1, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 192, padding=1, kernel_size=3)
        self.conv2_2 = nn.Conv2d(192, 192, padding=1, kernel_size=3, stride=2)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c1_2 = F.relu(self.conv1_2(c1))
        p1 = c1_2
        c2 = F.relu(self.conv2(p1))
        c2_2 = F.relu(self.conv2_2(c2))
        return self.variation_forward(c2_2)


class CCCNet(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        super(CCCNet, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 96, padding=1, kernel_size=3)
        self.conv1_2 = nn.Conv2d(96, 96, padding=1, kernel_size=3)
        self.conv1_3 = nn.Conv2d(96, 96, padding=1, kernel_size=3)

        self.conv2 = nn.Conv2d(96, 192, padding=1, kernel_size=3)
        self.conv2_2 = nn.Conv2d(192, 192, padding=1, kernel_size=3)
        self.conv2_3 = nn.Conv2d(192, 192, padding=1, kernel_size=3)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c1_2 = F.relu(self.conv1_2(c1))
        c1_3 = F.relu(self.conv1_3(c1_2))
        p1 = self.pool1(c1_3)
        c2 = F.relu(self.conv2(p1))
        c2_2 = F.relu(self.conv2_2(c2))
        c2_3 = F.relu(self.conv2_3(c2_2))
        p2 = self.pool2(c2_3)

        return self.variation_forward(p2)


class ACCNet(BaseNet):
    def __init__(self, num_classes=10, **kwargs):
        super(ACCNet, self).__init__(num_classes)
        self.conv1 = nn.Conv2d(3, 96, padding=1, kernel_size=3)
        self.conv1_2 = nn.Conv2d(96, 96, padding=1, kernel_size=3)
        self.conv1_3 = nn.Conv2d(96, 96, padding=1, kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 192, padding=1, kernel_size=3)
        self.conv2_2 = nn.Conv2d(192, 192, padding=1, kernel_size=3)
        self.conv2_3 = nn.Conv2d(192, 192, padding=1, kernel_size=3, stride=2)

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c1_2 = F.relu(self.conv1_2(c1))
        p1 = F.relu(self.conv1_3(c1_2))
        c2 = F.relu(self.conv2(p1))
        c2_2 = F.relu(self.conv2_2(c2))
        p2 = F.relu(self.conv2_3(c2_2))

        return self.variation_forward(p2)