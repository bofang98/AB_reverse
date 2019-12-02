import torch
import torch.nn as nn
import sys
sys.path.append("..")
from model.c3d import C3D
from model.net_part import *

class SSCN(nn.Module):
    def __init__(self, base_network,
                 with_classifier=False,
                 adaptive_contact=False,
                 num_classes=4):

        super(SSCN, self).__init__()

        self.base_network = base_network
        self.with_classifier = with_classifier
        self.adaptive_contact = adaptive_contact
        self.num_classes = num_classes

        self.conv6 = conv3d(512, 512)
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1))

        self.up1 = up3d(512, 256)
        self.up2 = up3d(256, 128)
        self.up3 = up3d(128, 64)

        self.up4_1 = nn.ConvTranspose3d(64,64,kernel_size=(1,2,2),stride=(1,2,2))

        self.outc = conv3d(64, 3)

        if self.with_classifier:
            self.fc7 = nn.Linear(512*2, 512)
            self.fc8 = nn.Linear(512, self.num_classes)
            self.relu = nn.ReLU(inplace=True)
            self.dropout = nn.Dropout(p=0.5)




    def forward(self, x1, x2):

        x1 = self.relu(self.base_network(x1))
        x2 = self.relu(self.base_network(x2))
        #print(x.shape)

        if self.with_classifier:

            h = torch.cat((x1, x2), dim=1)
            h = self.fc7(h)
            h = self.dropout(h)
            h = self.fc8(h)

        if self.with_classifier:
            return h
        else:
            return x2



if __name__ == '__main__':
    base = C3D(with_classifier=False)
    sscn = SSCN(base, with_classifier=True, num_classes=15)

    input_tensor = torch.autograd.Variable(torch.rand(4, 3, 16, 112, 112))
    out = sscn(input_tensor, input_tensor)
    print(out.shape)

