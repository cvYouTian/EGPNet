# -*-coding:utf-8-*-
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.utils.data as Data
from argparse import ArgumentParser
from tqdm import tqdm
import os.path as ops
import numpy as np
from data_utils import DatasetLoader
from metric import PD_FA, ROCMetric, mIoU
from metrics import mIoU
from sklearn.metrics import auc
# torch自带保存文件
from torchvision.utils import save_image
from PIL import Image
from scipy.io import savemat
from pathlib import Path


def parse_args():
    #
    # Setting parameters
    #
    parser = ArgumentParser(description='Implement of EGPNet model')
    parser.add_argument('--base_dir', type=str,
                        default='/home/youtian/Documents/pro/pyCode/CFFNet-V2/data/dataset/',
                        help='sirst_aug IRSTD-1k   MDvsFA_cGAN')
    parser.add_argument('--dataset', type=str, default='IRSTD-1k', help='sirst_aug IRSTD-1k   MDvsFA_cGAN')
    args = parser.parse_args()
    return args


def save_iamge_2(output, name1, name2):
    # output_dir = Path(args.output_dir)
    # if not output_dir.exists():
    #     output_dir.mkdir(parents=True, exist_ok=True)

    # tensor = torch.sigmoid(output)
    tensor = torch.sigmoid(output)
    if tensor.dim() == 4:
        tensor = output.squeeze(0)  # 去除batch维度 [B,C,H,W] -> [C,H,W]

    # 2. 转换通道顺序 PyTorch的[C,H,W] -> Matplotlib的[H,W,C]
    tensor = tensor.permute(1, 2, 0) if tensor.dim() == 3 else tensor

    # 3. 转换为CPU和numpy格式
    img = tensor.cpu().detach().numpy()

    # 4. 归一化处理
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
    elif img.dtype == torch.uint8:
        img = img.astype(np.uint8)  # 保持0-255范围
    print(f"save to {name2} \n")

    # 单独保存预测mask
    Image.fromarray((img.squeeze() * 255).astype(np.uint8)).save(name2)

class Main(object):
    def __init__(self, args):
        self.args = args

        valset = DatasetLoader(args, mode='test')
        self.val_data_loader = Data.DataLoader(valset, batch_size=1)

        print(len(self.val_data_loader))

        self.ROC_metric = ROCMetric(1, 20)
        self.PD_FA_metric = PD_FA(1, 100)
        self.mIoU_metric = mIoU(1)
        self.nIoU_metric = mIoU(1)

        folder_name = '/home/youtian/Documents/pro/pyCode/EGPNet/result/IRSTD-1k/checkpoint/Epoch-146_IoU-0.6080.pkl'
        path = '/result/' + self.args.dataset + '/checkpoint'
        self.load_pkl = ops.join(path, folder_name)
        self.net = torch.load(self.load_pkl)

        self.save_img = './result/' + self.args.dataset + '/predict/'
        self.save_label = './result/' + self.args.dataset + '/label/'
        if not ops.exists(self.save_img):
            os.mkdir(self.save_img)
        if not ops.exists(self.save_label):
            os.mkdir(self.save_label)

    def test(self):

        self.PD_FA_metric.reset()
        self.mIoU_metric.reset()
        self.ROC_metric.reset()
        self.nIoU_metric.reset()

        self.net.eval()

        tbar = tqdm(self.val_data_loader)

        for i, (data, labels, img_name) in enumerate(tbar):

            if data.shape[2] % 8 != 0:
                data = data[:, :, :-(data.shape[2] % 8), :]
                labels = labels[:, :, :-(labels.shape[2] % 8), :]

            if data.shape[3] % 8 != 0:
                data = data[:, :, :, :-(data.shape[3] % 8)]
                labels = labels[:, :, :, :-(labels.shape[3] % 8)]
            # perform test

            with torch.no_grad():
                # ???
                # output, edge_out = self.net(data.cuda(), data.cuda())
                output, edge_out = self.net(data.cuda())


                labels = labels[:, 0:1, :, :].cpu()
                output = output.cpu()

            output[output > 1.] = 1
            output[output < 0] = 0
            # self.ROC_metric.update(output, labels)
            # 在这个里进行了二值化.
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            self.mIoU_metric.update(output, labels)
            self.PD_FA_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)

            tpr, fpr, recall, precision = self.ROC_metric.get()
            _, mIOU = self.mIoU_metric.get()
            # FA, PD = self.PD_FA_metric.get(len(self.val_data_loader), labels)
            FA, PD = self.PD_FA_metric.get(len(self.val_data_loader))
            _, nIOU = self.nIoU_metric.get()

            auc_value = auc(fpr, tpr)
            tbar.set_description('mIoU:%f, nIOU:%f, AUC:%f, FA:%f, PD:%f'
                                 % (mIOU, nIOU, auc_value, FA[0] * 1000000, PD[0]))

            output = output[0, :]
            labels = labels[0, :]

            save_iamge_2(output, img_name, self.save_img + img_name[0])

            # save_image(output, self.save_img + img_name[0], normalize=True)
            # save_image(labels, self.save_label + img_name[0], normalize=True)
            # savemat('evulate_' + self.args.dataset + '_' + self.type + '.mat',
            #         {'miou': mIOU, 'prec': precision, 'recall': recall, 'auc': auc_value, 'tpr': tpr, 'fpr': fpr,
            #          'FA': FA[0] * 1000000, 'PD': PD[0]})


if __name__ == '__main__':
    args = parse_args()
    test_img = Main(args)
    test_img.test()
