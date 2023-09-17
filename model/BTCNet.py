import functools
from module_util import *
from Quantizer import *
from FCA import FCA


class RDFE(nn.Module):

    def __init__(self, nf=64, gc=32, bias=True):
        super(RDFE, self).__init__()

        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.silu = nn.SiLU(inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.silu(self.conv1(x))
        x2 = self.silu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.silu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.silu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class CAFE(nn.Module):

    def __init__(self, nf, gc=32):
        super(CAFE, self).__init__()
        self.RDFE1 = RDFE(nf, gc)
        self.RDFE2 = RDFE(nf, gc)
        self.RDFE3 = RDFE(nf, gc)
        self.FCA = FCA(nf)

    def forward(self, x):
        out = self.RDFE1(x)
        out = self.RDFE2(out)
        out = out * 0.2 + x
        out = self.RDFE3(out)
        ca = self.FCA(out)
        out = ca * out
        return out * 0.2 + x


class BTCNetDecoder(nn.Module):
    def make_layer(block, n_layers):
        layers = []
        for _ in range(n_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def __init__(self, in_nc, out_nc, nf, nb, gc=32, up_scale = 4):
        super(BTCNetDecoder, self).__init__()
        CAFE_block_f = functools.partial(CAFE, nf=nf, gc=gc)
        self.up_scale = up_scale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.CAFE_trunk = make_layer(CAFE_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)


        self.upconv1 = nn.Conv2d(nf, nf + gc, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf + gc, 2 * nf, 3, 1, 1, bias=True)
        self.upconv3 = nn.Conv2d(2 * nf, 2 * nf + gc, 3, 1, 1, bias=True)
        self.upconv4 = nn.Conv2d(2 * nf + gc, out_nc, 3, 1, 1, bias=True)
        self.silu = nn.SiLU(inplace=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.CAFE_trunk(fea))
        fea = fea + trunk

        # spatial/spectral SR
        if self.up_scale == 2:
            fea = self.silu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
            fea = self.silu(self.upconv2(fea))
        if self.up_scale == 4:
            fea = self.silu(self.upconv2(self.silu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))))
            fea = self.upconv4(self.silu(self.upconv3(F.interpolate(fea, scale_factor=2, mode='nearest'))))

        return fea


#  172*128*4=88064 --> 32*1*27 --> cr=1%
# 64*2*64 --> cr=9.30%
# 64*2*32 --> cr=4.65%
# 64*2*103--->cr=14.97%
# 64*2*140 -->cr=20.3%

class BTCNet(nn.Module):
    def __init__(self, snr=0, cr=1, bit_num=8):
        super(BTCNet, self).__init__()
        self.snr = snr

        if cr == 1:
            last_stride = 2
            last_ch = 27
            last_kernel_w = 1
            last_padding_w = 0
        else:
            last_stride = 1 
            last_kernel_w = 2
            last_padding_w = 1
            
        up_scale = 4 if cr<5 else 2
        if cr==5:
            last_ch = 32
        elif cr==10:
            last_ch = 64
        elif cr==15:
            last_ch=103
        elif cr==20:
            last_ch=140

        self.encoder = nn.Sequential(
            QConv(172, 128, [3, 3], stride=[2, 2], padding=[1, 0], num_w_bit=bit_num, num_a_bit=bit_num),
            nn.LeakyReLU(True),
            QConv(128, 64, [3, 1], stride=[1, 1], padding=[1, 0], num_w_bit=bit_num, num_a_bit=bit_num),
            nn.LeakyReLU(True),
            QConv(64, last_ch, [3, last_kernel_w], stride=[last_stride, 1], padding=[1, last_padding_w],
                    num_w_bit=bit_num, num_a_bit=bit_num),
            nn.LeakyReLU(True),
            QAct(num_a_bit=bit_num)
        )
        print(self.encoder)

        self.decoder = BTCNetDecoder(last_ch, 172, 64, 16, up_scale=up_scale)


    def awgn(self, x, snr):
        snr = 10**(snr/10.0)
        xpower = torch.sum(x**2)/x.numel()
        npower = torch.sqrt(xpower / snr)
        return x + torch.randn(x.shape).cuda() * npower


    def forward(self, data, mode=0):
        if mode==0:
            x = self.encoder(data)

            if self.snr > 0:
                x = self.awgn(x, self.snr)
            y = self.decoder(x)
            return y, x
        elif mode==1:
            return self.encoder(data)
        elif mode==2:
            return self.decoder(data)
        else:
            return self.decoder(self.encoder(data))
