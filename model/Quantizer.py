import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:1')


class RoundWithGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        delta = torch.max(x) - torch.min(x)
        x = (x / delta + 0.5)
        return x.round() * 2 - 1

    @staticmethod
    def backward(ctx, g):
        return g


class QConv(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
                 momentum=0.1,
                 num_w_bit=8, num_a_bit=8, QInput=True, bSetQ=True):
        super(QConv, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.num_w_bit = num_w_bit
        self.num_a_bit = num_a_bit
        self.quan_input = QInput
        self.bit_w_range = 2 ** self.num_w_bit - 1
        self.bit_a_range = 2 ** self.num_a_bit - 1
        self.is_quan = bSetQ
        self.momentum = momentum
        if self.is_quan:
            # Weight
            self.uW = nn.Parameter(data=torch.tensor(2 ** 8).float()).to(device)
            self.lW = nn.Parameter(data=torch.tensor(0).float()).to(device)
            self.register_buffer('running_uw', torch.tensor([self.uW.data]))  # init with uw
            self.register_buffer('running_lw', torch.tensor([self.lW.data]))  # init with lw
            self.betaW = nn.Parameter(data = torch.tensor([0.1], device=device).float(), requires_grad=True)
            # Bias
            if self.bias is not None:
                self.uB = nn.Parameter(data=torch.tensor(2 ** 8).float()).to(device)
                self.lB = nn.Parameter(data=torch.tensor(0).float()).to(device)
                self.register_buffer('running_uB', torch.tensor([self.uB.data]))  # init with ub
                self.register_buffer('running_lB', torch.tensor([self.lB.data]))  # init with lb
                self.betaB = nn.Parameter(data = torch.tensor([0.1], device=device).float(), requires_grad=True)

            # Activation input
            if self.quan_input:
                self.uA = nn.Parameter(data=torch.tensor(2 ** 8).float()).to(device)
                self.lA = nn.Parameter(data=torch.tensor(0).float()).to(device)
                self.register_buffer('running_uA', torch.tensor([self.uA.data]))  # init with uA
                self.register_buffer('running_lA', torch.tensor([self.lA.data]))  # init with lA
                self.betaA = nn.Parameter(data = torch.tensor([0.1], device=device).float(), requires_grad=True)

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def phi_function(self, x, mi, beta, delta):
        # beta should less than 2 or log will be None
        beta = torch.where(beta >= 2.0, torch.tensor([2.0]).to(device), beta).to(device)
        s = 1 / (1 - beta)
        k = (2 / beta - 1).log() * (1 / delta)
        x = (((x - mi) * k).tanh()) * s
        return x

    def sgn(self, x):
        x = RoundWithGradient.apply(x)

        return x

    def dequantize(self, x, lower_bound, delta, interval):

        # save mem
        x = ((x + 1) / 2 + interval) * delta + lower_bound

        return x

    def forward(self, x):
        if self.is_quan:
            # Weight Part
            # moving average
            if self.training:
                cur_running_lw = self.running_lw.mul(1 - self.momentum).add((self.momentum) * self.lW)
                cur_running_uw = self.running_uw.mul(1 - self.momentum).add((self.momentum) * self.uW)
            else:
                cur_running_lw = self.running_lw
                cur_running_uw = self.running_uw

            # limit the value of beta
            if self.betaW.item() < 0.05:
                self.betaW.requires_grad = False
            if self.betaW.item() > 0.2:
                self.betaW.requires_grad = False

            Qweight = self.clipping(self.weight, cur_running_uw, cur_running_lw)
            cur_max = torch.max(Qweight)  # determine the u
            cur_min = torch.min(Qweight)  # determine the l
            delta = (cur_max - cur_min) / (self.bit_w_range)
            interval = (((Qweight - cur_min) / delta).trunc()).to(device)
            mi = (interval + 0.5) * delta + cur_min
            Qweight = self.phi_function(Qweight, mi, self.betaW, delta).to(device)
            Qweight = self.sgn(Qweight).to(device)
            Qweight = self.dequantize(Qweight, cur_min, delta, interval).to(device)

            Qbias = self.bias
            # Bias
            if self.bias is not None:
                # self.running_lB.mul_(1-self.momentum).add_((self.momentum) * self.lB)
                # self.running_uB.mul_(1-self.momentum).add_((self.momentum) * self.uB)
                if self.training:
                    cur_running_lB = self.running_lB.mul(1 - self.momentum).add((self.momentum) * self.lB)
                    cur_running_uB = self.running_uB.mul(1 - self.momentum).add((self.momentum) * self.uB)
                else:
                    cur_running_lB = self.running_lB
                    cur_running_uB = self.running_uB

                Qbias = self.clipping(self.bias, cur_running_uB, cur_running_lB).to(device)
                cur_max = torch.max(Qbias)
                cur_min = torch.min(Qbias).to(device)
                delta = ((cur_max - cur_min) / (self.bit_w_range)).to(device)
                interval = (((Qbias - cur_min) / delta).trunc()).to(device)
                mi = ((interval + 0.5) * delta + cur_min).to(device)
                Qbias = self.phi_function(Qbias, mi, self.betaB, delta)
                Qbias = self.sgn(Qbias)
                Qbias = self.dequantize(Qbias, cur_min, delta, interval)

            # Input(Activation)
            Qactivation = x
            if self.quan_input:

                if self.training:
                    cur_running_lA = self.running_lA.mul(1 - self.momentum).add((self.momentum) * self.lA)
                    cur_running_uA = self.running_uA.mul(1 - self.momentum).add((self.momentum) * self.uA)
                else:
                    cur_running_lA = self.running_lA
                    cur_running_uA = self.running_uA

                if self.betaA.item() < 0.05:  # limit the value of beta
                    self.betaA.requires_grad = False
                if self.betaA.item() > 0.2:
                    self.betaA.requires_grad = False

                Qactivation = self.clipping(x, cur_running_uA, cur_running_lA).to(device)
                cur_max = torch.max(Qactivation)
                cur_min = torch.min(Qactivation).to(device)
                delta = ((cur_max - cur_min) / (self.bit_a_range)).to(device)
                interval = (((Qactivation - cur_min) / delta).trunc()).to(device)
                mi = ((interval + 0.5) * delta + cur_min).to(device)
                Qactivation = self.phi_function(Qactivation, mi, self.betaA, delta).to(device)
                Qactivation = self.sgn(Qactivation).to(device)
                Qactivation = self.dequantize(Qactivation, cur_min, delta, interval).to(device)

            output = F.conv2d(Qactivation, Qweight, Qbias, self.stride, self.padding, self.dilation, self.groups)

        else:
            output = F.conv2d(x, self.weight, self.bias, self.stride,
                              self.padding, self.dilation, self.groups)

        return output


class QAct(nn.Module):
    def __init__(self, num_a_bit=8, momentum=0.1, quan_input=True):
        super(QAct, self).__init__()
        self.num_a_bit = num_a_bit
        self.bit_a_range = 2 ** self.num_a_bit - 1
        self.momentum = momentum
        # Activation input
        if quan_input:
            self.uA = nn.Parameter(data=torch.tensor(2 ** 31 - 1).float()).to(device)
            self.lA = nn.Parameter(data=torch.tensor((-1) * (2 ** 32)).float()).to(device)
            self.register_buffer('running_uA', torch.tensor([self.uA.data]))  # init with uA
            self.register_buffer('running_lA', torch.tensor([self.lA.data]))  # init with lA
            self.betaA = nn.Parameter(data = torch.tensor([0.1], device=device).float(), requires_grad=True)

    def clipping(self, x, upper, lower):
        # clip lower
        x = x + F.relu(lower - x)
        # clip upper
        x = x - F.relu(x - upper)

        return x

    def phi_function(self, x, mi, beta, delta):
        # beta should less than 2 or log will be None
        s = 1 / (1 - beta)
        k = (2 / beta - 1).log() * (1 / delta)
        x = (((x - mi) * k).tanh()) * s
        return x

    def sgn(self, x):
        x = RoundWithGradient.apply(x)

        return x

    def dequantize(self, x, lower_bound, delta, interval):
        # save mem
        x = ((x + 1) / 2 + interval) * delta + lower_bound

        return x

    def forward(self, x):
        # Input(Activation)
        if self.training:
            cur_running_lA = self.running_lA.mul(1 - self.momentum).add((self.momentum) * self.lA)
            cur_running_uA = self.running_uA.mul(1 - self.momentum).add((self.momentum) * self.uA)
        else:
            cur_running_lA = self.running_lA
            cur_running_uA = self.running_uA

        Qactivation = self.clipping(x, cur_running_uA, cur_running_lA).to(device)
        cur_max = torch.max(Qactivation)
        cur_min = torch.min(Qactivation).to(device)
        delta = ((cur_max - cur_min) / (self.bit_a_range)).to(device)
        interval = (((Qactivation - cur_min) / delta).trunc()).to(device)
        mi = ((interval + 0.5) * delta + cur_min).to(device)

        if self.betaA.item() < 0.05:  # limit the value of beta
            self.betaA.requires_grad = False
        if self.betaA.item() > 0.2:
            self.betaA.requires_grad = False

        Qactivation = self.phi_function(Qactivation, mi, self.betaA, delta).to(device)
        Qactivation = self.sgn(Qactivation).to(device)
        Qactivation = self.dequantize(Qactivation, cur_min, delta, interval).to(device)
        output = Qactivation

        return output
