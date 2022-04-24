import torch
import random
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import torchvision.transforms as transforms
from args import get_args
from utils import weight_init
from dataset import FaceDataset
from model import Generator, Discriminator
from model_w import Generator_w, Discriminator_w

if __name__ == '__main__':
    args = get_args()
    args.manualSeed = 99
    print("Random Seed:", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    nc = 3
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    net_G = Generator(ngpu=args.ngpu, nz=args.nz, nc=nc, ngf=args.ngf).to(device)  # 初始化生成器
    net_G.load_state_dict(torch.load('checkpoint/netG_epoch_10.pth', map_location=torch.device('cpu')))
    fixed_noise = torch.rand(args.batchSize, args.nz, 1, 1, device=device)  # z
    print(fixed_noise.shape)
    fake = net_G(fixed_noise)
    vutils.save_image(fake.detach(),
                      '%s/fake_samples_epoch_test.png' % (args.outf),
                      normalize=True)


