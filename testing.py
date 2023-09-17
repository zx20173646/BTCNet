import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import *
from dataset import *
from trainOps import *
from Huffman_BTCNet import *
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from BTCNet import BTCNet

# torch.backends.cudnn.benchmark=True
# Hyperparameters
batch_size = 8
device = 'cuda:1' ## cpu or cuda (set cuda if gpu avaiilable)
VAL_HR = 256
INTERVAL= 4
WIDTH=4
BANDS = 172
CR = 1        ## CR = 1, 5, 10, 15, 20
SIGMA = 0     ## Noise free -> SIGMA = 0.0
              ## Noise mode -> SIGMA > 0.0
TARGET = '4fig'
SNR = 50

prefix = 'BTC-Net'

if not os.path.isdir('Rec'):
    os.mkdir('Rec')
if not os.path.isdir('HuffmanLog'):
    os.mkdir('HuffmanLog')

testdata = loadTxt('testpath/test.txt')


## Setup the dataloader
val_loader = torch.utils.data.DataLoader(dataset_h5(testdata, mode='Validation',root=''), batch_size=10, shuffle=False, pin_memory=False)

model = BTCNet(cr=CR, bit_num=10).to(device)

state = torch.load('checkpoint/BTC-Net_1065subimg_118subimg_cr_1_epoch_2000.pth', map_location='cuda:1')

from collections import OrderedDict
# new_state_dict = OrderedDict()
# # load baseline static
# for k, v in DAQstate.items():
#     name = k[7:] # remove `module.`
#     new_state_dict[name] = v
# load params

model.load_state_dict(state)

with torch.no_grad():
    huffman, rmses, sams, fnames, psnrs, time1, time2 = [], [], [], [], [], [], []
    for ind2, (vx, vfn) in enumerate(val_loader):
        model.eval()
        vx = vx.view(vx.size()[0]*vx.size()[1], vx.size()[2], vx.size()[3], vx.size()[4])
        vx= vx.to(device).permute(0,3,1,2).float()
        if SIGMA>0:
            val_dec = model(awgn(model(vx, mode=1), SNR), mode=2)

        else:
            # start_time = time.time()

            val_dec, _ = model(vx)

            data = _.cpu().numpy()
            data = np.float32(data)
            data = np.reshape(data, (data.shape[0] * data.shape[1], -1))

            HuffmanCR = huffman_encoder_BTCNet(data)
            huffman.append(HuffmanCR)

        val_batch_size = len(vfn)
        img = [np.zeros((VAL_HR, VAL_HR, BANDS)) for _ in range(val_batch_size)]
        val_dec = val_dec.permute(0,2,3,1).cpu().numpy()
        cnt = 0

        quality_list = []
        searches = []

        for bt in range(val_batch_size):
            for z in range(0, VAL_HR, INTERVAL):
                img[bt][:,z:z+WIDTH,:] = val_dec[cnt]
                cnt +=1
            save_path = vfn[bt].split('/')
            save_path = save_path[-2] + '-' + save_path[-1]
            np.save('Rec/%s.npy' % (save_path), img[bt])

            GT = lmat(vfn[bt]).astype(np.float32)
            maxv, minv=np.max(GT), np.min(GT)
            img[bt] = img[bt]*(maxv-minv) + minv ## De-normalization
            sams.append(sam(img[bt], GT))
            psnrs.append(psnr(img[bt], GT))
            rmses.append(np.sqrt(np.mean((GT-img[bt])**2)))
            fnames.append(save_path)
            print('{:25} '.format(vfn[bt].split('/')[-1])+' %.3f/%.3f/%.3f' %
            (sams[bt], rmses[bt], psnrs[bt]))

    print('\n\n %.3f/%.3f/%.3f,  huffmanCR:%.3f ' %
          (np.mean(sams), np.mean(rmses), np.mean(psnrs), np.mean(huffman)))
    # print('%.3f / %.3f / %.3f' %(np.mean(psnrs), np.mean(rmses), np.mean(sams)))
    with open('HuffmanLog/huffmanlog.txt', 'a') as f:
        f.write('model: %s, val-RMSE: %.3f, val-SAM: %.3f, psnr:%.3f, HuffmanCR: %.3f, BTCNetEncodetime:%.3f \n' %
          (prefix, np.mean(rmses), np.mean(sams), np.mean(psnrs), np.mean(huffman), np.mean(time2) ))


