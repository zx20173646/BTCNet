import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch.utils.data as pydata
# from torchsummary import summary
import numpy as np
import scipy.io as sio
import random
from matplotlib import cm
import spectral as spy
from sklearn import metrics
from sklearn import preprocessing
import time
from torch.autograd import Variable
from H_datapy import *
import scipy.io as sio
from Classifier import *
import torch.nn.functional as F
from autis import *

samples_type=['ratio','same_num'][0]
mode = '2'   # two modes: train : 1, test : 2
fgsm = 0    # fast gradient sign method : 1, else: 0
save_path = 'checkpointIP/CNet_on_%s128_OA%s'  # the prefix of saved trained checkpoint
state_dict = torch.load('checkpointIP/CNet_on_indian128_OA0.9471253190713504')  # trained checkpoint


for (FLAG,curr_train_ratio) in [(1, 0.1)]:   # train sample ratio: 10%
    OA_ALL = []
    AA_ALL = []
    KPP_ALL = []
    AVG_ALL = []
    # trained checkpoint seed: 0, 2, 3, 4, 5, 7, 9, 11, 12, 14
    Seed = range(0, 1, 20)   # the range of random seed
    #  The random seeds corresponding to the 10 trained checkpoints are recorded in logIP/IP_ori.txt
    #  In the annotation # after each experimental result.
    Seed_List = [0]   # current seed value

    if FLAG == 1:
        data_mat = sio.loadmat('./Datasets/Indianpines/indian_pines128.mat')  # original data
        # data_mat = sio.loadmat('./Datasets/Indianpines/indian_pines128-BTC-Net.mat')  # Rec data by BTC-Net
        # data_mat = sio.loadmat('./Datasets/Indianpines/indian_pines128-DCSN.mat')
        # data_mat = sio.loadmat('./Datasets/Indianpines/indian_pines128-PCA+JPEG2K.mat')
        # data_mat = sio.loadmat('Datasets/Indianpines/indian_pines128-E3DTV.mat')
        data = data_mat['data']
        gt_mat = sio.loadmat('./Datasets/Indianpines/indian_pines128gt.mat')   # ground truth
        gt = gt_mat['gt']

        val_ratio = 0
        class_count = 16
        learning_rate = 5e-4
        weight_decay = 2e-5
        max_epoch = 600
        split_height = 1
        split_width = 1
        dataset_name = "indian"
        pass

    train_samples_per_class=curr_train_ratio
    val_samples=class_count


    train_ratio=curr_train_ratio
    if split_height == split_width == 1:
            EDGE = 0
    else:
            EDGE = 5

    cmap = cm.get_cmap('jet', class_count + 1)
    plt.set_cmap(cmap)
    m, n, d = data.shape
    n_bands=d

    data = np.reshape(data, [m * n, d])
    minMax = preprocessing.StandardScaler()
    data = minMax.fit_transform(data)
    data = np.reshape(data, [m, n, d])


    for curr_seed in Seed_List:
        random.seed(curr_seed)
        gt_reshape = np.reshape(gt, [-1])
        train_rand_idx = []
        val_rand_idx = []

        if samples_type=='ratio':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                rand_list = [i for i in range(samplesCount)]
                rand_idx = random.sample(rand_list, np.ceil(samplesCount * train_ratio).astype('int32'))  # 向上取整随机数
                rand_real_idx_per_class = idx[rand_idx]
                train_rand_idx.append(rand_real_idx_per_class)
            train_rand_idx = np.array(train_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)
            print(np.sum(train_data_index))

            train_data_index = set(train_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)


            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)

            test_data_index = all_data_index - train_data_index - background_idx


            val_data_count = int(val_ratio * (len(test_data_index) + len(train_data_index)))
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)
            test_data_index = test_data_index - val_data_index


            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)

        if samples_type=='same_num':
            for i in range(class_count):
                idx = np.where(gt_reshape == i + 1)[-1]
                samplesCount = len(idx)
                real_train_samples_per_class=train_samples_per_class
                rand_list = [i for i in range(samplesCount)]
                if real_train_samples_per_class>=samplesCount:
                    #real_train_samples_per_class=samplesCount
                    real_train_samples_per_class=int(train_samples_per_class/2)
                    # val_samples_per_class=0
                rand_idx = random.sample(rand_list,
                                         real_train_samples_per_class)
                rand_real_idx_per_class_train = idx[rand_idx[0:real_train_samples_per_class]]
                train_rand_idx.append(rand_real_idx_per_class_train)
                # if val_samples_per_class>0:
                #     rand_real_idx_per_class_val = idx[rand_idx[-val_samples_per_class:]]
                #     val_rand_idx.append(rand_real_idx_per_class_val)
            train_rand_idx = np.array(train_rand_idx)
            val_rand_idx = np.array(val_rand_idx)
            train_data_index = []
            for c in range(train_rand_idx.shape[0]):
                a = train_rand_idx[c]
                for j in range(a.shape[0]):
                    train_data_index.append(a[j])
            train_data_index = np.array(train_data_index)


            train_data_index = set(train_data_index)
            # val_data_index = set(val_data_index)
            all_data_index = [i for i in range(len(gt_reshape))]
            all_data_index = set(all_data_index)


            background_idx = np.where(gt_reshape == 0)[-1]
            background_idx = set(background_idx)
            test_data_index = all_data_index - train_data_index - background_idx

            val_data_count = int(val_samples)
            val_data_index = random.sample(test_data_index, val_data_count)
            val_data_index = set(val_data_index)

            test_data_index=test_data_index-val_data_index

            test_data_index = list(test_data_index)
            train_data_index = list(train_data_index)
            val_data_index = list(val_data_index)


        train_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(train_data_index)):
            train_samples_gt[train_data_index[i]] = gt_reshape[train_data_index[i]]
            pass
        Train_Label=np.reshape(train_samples_gt, [m,n])


        test_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(test_data_index)):
            test_samples_gt[test_data_index[i]] = gt_reshape[test_data_index[i]]
            pass

        Test_Label = np.reshape(test_samples_gt, [m, n])  # 测试样本图


        val_samples_gt = np.zeros(gt_reshape.shape)
        for i in range(len(val_data_index)):
            val_samples_gt[val_data_index[i]] = gt_reshape[val_data_index[i]]
            pass
        Val_Label=np.reshape(val_samples_gt,[m,n])


        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        train_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(train_samples_gt.shape[0]):
            class_idx = train_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                train_samples_gt_vector[i] = temp
        train_samples_gt_vector = np.reshape(train_samples_gt_vector, [m, n, class_count])

        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        test_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(test_samples_gt.shape[0]):
            class_idx = test_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                test_samples_gt_vector[i] = temp
        test_samples_gt_vector = np.reshape(test_samples_gt_vector, [m, n, class_count])

        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        val_samples_gt_vector = np.zeros([m * n, class_count], np.float)
        for i in range(val_samples_gt.shape[0]):
            class_idx = val_samples_gt[i]
            if class_idx != 0:
                temp = np.zeros([class_count])
                temp[int(class_idx - 1)] = 1
                val_samples_gt_vector[i] = temp
        val_samples_gt_vector = np.reshape(val_samples_gt_vector, [m, n, class_count])


        train_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        train_samples_gt = np.reshape(train_samples_gt, [m * n])
        for i in range(m * n):
            if train_samples_gt[i] != 0:
                train_label_mask[i] = temp_ones
        train_label_mask = np.reshape(train_label_mask, [m, n, class_count])


        test_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        test_samples_gt = np.reshape(test_samples_gt, [m * n])
        for i in range(m * n):
            if test_samples_gt[i] != 0:
                test_label_mask[i] = temp_ones
        test_label_mask = np.reshape(test_label_mask, [m, n, class_count])


        val_label_mask = np.zeros([m * n, class_count])
        temp_ones = np.ones([class_count])
        val_samples_gt = np.reshape(val_samples_gt, [m * n])
        for i in range(m * n):
            if val_samples_gt[i] != 0:
                val_label_mask[i] = temp_ones
        val_label_mask = np.reshape(val_label_mask, [m, n, class_count])


        # t1=Train_Label
        # t1[Train_Label>0]=1
        # num=t1.sum()
        # t2=Test_Label
        # t2[Test_Label>0]=1
        # num2=t2.sum()
        Train_Split_Data, Train_Split_GT = SpiltHSI(data, Train_Label, [split_height, split_width], EDGE)
        Test_Split_Data, Test_Split_GT = SpiltHSI(data, Test_Label, [split_height, split_width], EDGE)
        _, patch_height, patch_width, bands = Train_Split_Data.shape
        patch_height -= EDGE * 2
        patch_width -= EDGE * 2

        zero_vector = np.zeros([class_count])
        all_label_mask = np.ones([1, m, n, class_count])


    train_h=HData((np.transpose(Train_Split_Data,(0,3,1,2)).astype("float32"), Train_Split_GT), None)
    test_h=HData((np.transpose(Test_Split_Data,(0,3,1,2)).astype("float32"), Test_Split_GT), None)
    trainloader=torch.utils.data.DataLoader(train_h, shuffle=True)
    testloader=torch.utils.data.DataLoader(test_h, shuffle=True)

    use_cuda = torch.cuda.is_available()

    model = ClassifierIP(class_count, n_bands, 150)


    print(model)
    if use_cuda: torch.backends.cudnn.benchmark = True
    if use_cuda: model.cuda()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-5)
    print('lr: ',learning_rate, '  weight_dacay: ', weight_decay)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.003, momentum=0.9, weight_decay=1e-4, nesterov=True)


    best_acc = -1

    if mode == '1':  # train mode

        for eep in range(max_epoch):
            for batch_idx, (inputs, labels) in enumerate(trainloader):
                if use_cuda:
                    inputs, labels = inputs.cuda(), labels.cuda()
                inputs, labels = torch.autograd.Variable(inputs), torch.autograd.Variable(labels)
                optimizer.zero_grad()
                output = model(inputs)
                # print(output.size(), labels.size())

                loss = criterion(output, labels.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if eep % 5 == 0:
                Output = []
                for Testbatch_idx, (Testinputs, Testtargets) in enumerate(
                        testloader):
                    if use_cuda:
                        Testinputs, Testtargets = Testinputs.cuda(), Testtargets.cuda()
                    Testinputs, Testtargets = torch.autograd.Variable(Testinputs), torch.autograd.Variable(Testtargets)
                    Testoutput = model(Testinputs)
                    Testoutput = Testoutput.data.cpu().numpy()
                    Testoutput = np.transpose(Testoutput, (0, 2, 3, 1))
                    Output.append(Testoutput[0])

                OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE,
                                         class_count + 1)
                AC, OA, AA, _, _ = ClassificationAccuracy(OutputWhole, Test_Label, class_count + 1)
                kappa = Kappa(OutputWhole, Test_Label, class_count + 1)
                print("eep", eep, " test", "OA", OA, "AA", AA, "kappa", kappa)
                print(AC)
            if eep == 260:
                OA = np.round(OA * 100, decimals=2)
                OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE,
                                         class_count + 1)
                Draw_Classification_Map(OutputWhole,
                                        'ResultsImage/' + dataset_name + '_' + str(train_ratio) + '_' + str(OA))

            if eep == (max_epoch - 5):
                torch.save(model.state_dict(), save_path % (dataset_name, str(OA)))

            if loss.data <= 0.00005:
                break

    elif mode == '2':  # test mode

        model.load_state_dict(state_dict)

        Output = []
        Output_FGSM = []
        for Testbatch_idx, (Testinputs, Testtargets) in enumerate(
                testloader):
            if use_cuda:
                Testinputs, Testtargets = Testinputs.cuda(), Testtargets.cuda()
            Testinputs, Testtargets = torch.autograd.Variable(Testinputs), torch.autograd.Variable(Testtargets)
            Testoutput = model(Testinputs)
            Testoutput = Testoutput.data.cpu().numpy()
            Testoutput = np.transpose(Testoutput, (0, 2, 3, 1))
            Output.append(Testoutput[0])

        OutputWhole = PatchStack(Output, m, n, patch_height, patch_width, split_height, split_width, EDGE,
                                 class_count + 1)
        AC, OA, AA, _, _ = ClassificationAccuracy(OutputWhole, Test_Label, class_count + 1)
        sio.savemat('Datasets/Indianpines/PredictionRecIndianPines.mat', {'gt': OutputWhole})
        Draw_Classification_Map(OutputWhole,
                                'ResultsImage/' + dataset_name + '_' + str(train_ratio) + '_' + str(OA))
        kappa = Kappa(OutputWhole, Test_Label, class_count + 1)
        print(" test", "OA", OA, "AA", AA, "kappa", kappa)
        print(AC)


        # fgsm
        if fgsm == 1:

            for Testbatch_idx, (Testinputs, Testtargets) in enumerate(
                    testloader):
                if use_cuda:
                    Testinputs, Testtargets = Testinputs.cuda(), Testtargets.cuda()
                Testinputs, Testtargets = torch.autograd.Variable(Testinputs, requires_grad=True), \
                    torch.autograd.Variable(Testtargets, requires_grad=False)
                Testoutput = model(Testinputs)
                Testoutput = Testoutput.data.cpu().numpy()
                Testoutput = np.transpose(Testoutput, (0, 2, 3, 1))
                Output.append(Testoutput[0])

            output = model(Testinputs)
            loss_val = criterion(output, Testtargets.long())
            loss_val.backward(retain_graph=True)
            grad = torch.sign(Testinputs.grad.data)
            adversarial_input = Testinputs.data + 0.01 * grad
            adversarial_output = model(adversarial_input)
            adversarial_output = adversarial_output.data.cpu().numpy()
            adversarial_output = np.transpose(adversarial_output, (0, 2, 3, 1))
            Output_FGSM.append(adversarial_output[0])

            OutputWhole_FGSM = PatchStack(Output_FGSM, m, n, patch_height, patch_width, split_height, split_width, EDGE,
                                          class_count + 1)

            AC, OA, AA, _, _ = ClassificationAccuracy(OutputWhole_FGSM, Test_Label, class_count + 1)
            kappa = Kappa(OutputWhole_FGSM, Test_Label, class_count + 1)
            print(" test_fgsm", "OA", OA, "AA", AA, "kappa", kappa)
            print(OA, AA, kappa, AC)

    model.train()
    model.eval()








