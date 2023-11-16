import torch
import torch.nn as nn
from torch.nn import Softmax
import math
#from math import round
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class Classifier1(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(Classifier1, self).__init__()
        self.bands=n_bands
        chanel=chanel
        kernel=5
        CCChannel=25

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()

        self.b2=nn.BatchNorm2d(self.bands+chanel)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=2, groups=CCChannel, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)
        self.con4=nn.Conv2d(CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=2, groups=chanel, bias=True)
        self.sd5=nn.Sigmoid()

        self.con6=nn.Conv2d(CCChannel+chanel, num_classes+1, 1, padding=0, bias=True)

    def forward(self, x):
        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)

        out4=self.b4(out2)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)

        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)

        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6


class ClassifierIP(nn.Module):
    def __init__(self, num_classes, n_bands, chanel):
        super(ClassifierIP, self).__init__()
        self.bands=n_bands
        chanel=chanel
        kernel=3
        CCChannel=25

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=1, groups=1, bias=True)
        self.sd1=nn.Sigmoid()

        self.b2=nn.BatchNorm2d(self.bands+chanel)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=1, groups=1, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)
        self.con4=nn.Conv2d(CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=1, groups=1, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=1, groups=1, bias=True)
        self.sd5=nn.Sigmoid()

        self.con6=nn.Conv2d(CCChannel+chanel, num_classes+1, 1, padding=0, bias=True)

    def forward(self, x):
        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)

        out4=self.b4(out2)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)

        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)

        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6


class ClassifierS(nn.Module):   # S
    def __init__(self, num_classes, n_bands, chanel):
        super(ClassifierS, self).__init__()
        self.bands=n_bands
        chanel=chanel
        kernel=3
        CCChannel=25

        self.b1=nn.BatchNorm2d(self.bands)
        self.con1=nn.Conv2d(self.bands, chanel, 1, padding=0,bias=True)
        self.s1=nn.Sigmoid()
        self.cond1=nn.Conv2d(chanel, chanel, kernel, padding=1, groups=chanel, bias=True)
        self.sd1=nn.Sigmoid()

        self.b2=nn.BatchNorm2d(self.bands+chanel)
        self.con2=nn.Conv2d(self.bands+chanel, chanel, 1, padding=0,bias=True)
        self.s2=nn.Sigmoid()
        self.cond2=nn.Conv2d(chanel, CCChannel, kernel, padding=1, groups=5, bias=True)
        self.sd2=nn.Sigmoid()

        self.b4=nn.BatchNorm2d(CCChannel)
        self.con4=nn.Conv2d(CCChannel, chanel, 1, padding=0, bias=True)
        self.s4=nn.Sigmoid()
        self.cond4=nn.Conv2d(chanel, chanel, kernel, padding=1, groups=chanel, bias=True)
        self.sd4=nn.Sigmoid()

        self.b5=nn.BatchNorm2d(CCChannel+chanel)
        self.con5=nn.Conv2d(CCChannel+chanel, chanel, 1, padding=0, bias=True)
        self.s5=nn.Sigmoid()
        self.cond5=nn.Conv2d(chanel, chanel, kernel, padding=1, groups=15, bias=True)
        self.sd5=nn.Sigmoid()

        self.con6=nn.Conv2d(CCChannel+chanel, num_classes+1, 1, padding=0, bias=True)

    def forward(self, x):
        out1=self.b1(x)
        out1=self.con1(out1)
        out1=self.s1(out1)
        out1=self.cond1(out1)
        out1=self.sd1(out1)

        out2=torch.cat((out1,x),1)
        out2=self.b2(out2)
        out2=self.con2(out2)
        out2=self.s2(out2)
        out2=self.cond2(out2)
        out2=self.sd2(out2)

        out4=self.b4(out2)
        out4=self.con4(out4)
        out4=self.s4(out4)
        out4=self.cond4(out4)
        out4=self.sd4(out4)

        out5=torch.cat((out4,out2),1)
        out5=self.b5(out5)

        out5=self.con5(out5)
        out5=self.s5(out5)
        out5=self.cond5(out5)
        out5=self.sd5(out5)

        out6=torch.cat((out5,out2),1)
        out6=self.con6(out6)

        return out6





