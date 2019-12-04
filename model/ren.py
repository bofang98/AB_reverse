import torch
import sys
sys.path.append("..")
from model.c3d import C3D
from model.net_part import *

class REN(nn.Module):
    def __init__(self,
                 base_network,
                 with_classifier=False,
                 num_classes=15):
        super(REN, self).__init__()

        self.base_network = base_network
        self.with_classifier = with_classifier
        self.num_classes = num_classes

        self.pool = nn.AdaptiveAvgPool3d(1)
        if self.with_classifier:
            self.fc6 = nn.Linear(512, self.num_classes)



    def forward(self, x):
        x = self.base_network(x)
        x = self.pool(x)
        x = x.view(-1, 512)

        if self.with_classifier:
            x = self.fc6(x)

        return x


if __name__ == '__main__':
    base = C3D(with_classifier=False);
    ren = REN(base, with_classifier=True, num_classes=101);

    input_tensor = torch.autograd.Variable(torch.rand(4, 3, 16, 112, 112))
    # print(input_tensor)
    out = ren(input_tensor)

    print(out.shape)

