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

def train(args):

    # Random Seed
    if args.manualSeed is None:
        args.manualSeed = 99 # random.randint(1, 10000)
    print("Random Seed:", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    transform = transforms.Compose([    # 数据集转换格式
        transforms.Resize(args.imageSize),
        transforms.CenterCrop(args.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FaceDataset(root = args.dataroot, transform = transform)  # 初始化数据集
    dataloader = DataLoader(dataset, batch_size = args.batchSize, shuffle = True, num_workers = args.workers)   # 加载数据集
    nc = 3

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu") # 选用设备

    net_G = Generator(ngpu = args.ngpu, nz = args.nz, nc = nc, ngf = args.ngf).to(device)
    if (device.type == 'cuda') and (args.ngpu > 1):
        net_G = nn.parallel(net_G, list(range(args.ngpu)))
    net_G.apply(weight_init)    # 生成器应用权值
    net_G.load_state_dict(torch.load('checkpoint/netG_epoch_10.pth', map_location=torch.device('cpu')))

    net_D = Discriminator(ngpu = args.ngpu, nc = nc, ndf = args.ndf).to(device)
    if (device.type == 'cuda') and (args.ngpu > 1):
        net_D = nn.parallel(net_D, list(range(args.ngpu)))
    net_D.apply(weight_init)    # 判别器应用权值
    net_D.load_state_dict(torch.load('checkpoint/netD_epoch_10.pth', map_location=torch.device('cpu')))

    criterion = nn.BCELoss()    # 使用损失函数
    fixed_noise = torch.rand(args.batchSize, args.nz, 1, 1, device = device) # z，输入噪声
    real_label = 1. # 图片标签
    fake_label = 0.

    optimizer_D = optim.Adam(net_D.parameters(), lr = args.lr, betas = (args.beta1, 0.999)) # 优化器设置
    optimizer_G = optim.Adam(net_G.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    G_losses = []   # 损失
    D_losses = []

    print("开始训练")  # 训练
    for epoch in range(args.epochs + 100):    # 训练循环次数
        net_D.train()   # 对判别器进行训练
        net_G.train()   # 对生成器进行训练
        for step, data in enumerate(dataloader, 0):
            # 更新D网络
            # maximize log(D(x)) + log(1 - D(G(z)))
            net_D.zero_grad()   # 梯度置零

            # Take all real batch
            real = data.to(device)  # 变量copy一份到device所指定的GPU上去
            b_size = real.size(0)   # 赋值batchsize
            label = torch.full((b_size, ), real_label, dtype = torch.float, device = device)
            output = net_D(real).view(-1)
            errD_real = criterion(output, label)    # 损失
            errD_real.backward()    # 反向传播
            D_x = output.mean().item()  # 平均值

            # Take all fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = net_G(noise)
            label.fill_(fake_label)
            output = net_D(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake        #
            optimizer_D.step()  # 更新优化器

            # 更新G网络
            # maximize log(D(G(z)))
            net_G.zero_grad()
            label.fill_(real_label)
            output = net_D(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_G.step()

            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, args.epochs, step, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if step % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % (args.outf),
                                  normalize = True)
                fake = net_G(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                                  normalize = True)

            G_losses.append(errG.item())
            D_losses.append(errD.item())
        if epoch % 5 == 0:
            torch.save(net_G.state_dict(), '%s/netG_epoch_%d.pth' % (args.ckpt, epoch))
            torch.save(net_D.state_dict(), '%s/netD_epoch_%d.pth' % (args.ckpt, epoch))

    plt.figure(figsize = (10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label = "G")
    plt.plot(D_losses, label = "D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train_w(args):

    # Random Seed
    if args.manualSeed is None:
        args.manualSeed = 99    # random.randint(1, 10000)
    print("Random Seed:", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    transform = transforms.Compose([    # 数据集进行处理
        transforms.Resize(args.imageSize),
        transforms.CenterCrop(args.imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = FaceDataset(root = args.dataroot, transform = transform)
    dataloader = DataLoader(dataset, batch_size = args.batchSize, shuffle = True, num_workers = args.workers)
    nc = 3

    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")

    net_G = Generator_w(ngpu = args.ngpu, nz = args.nz, nc = nc, ngf = args.ngf).to(device)   # 初始化生成器

    if (device.type == 'cuda') and (args.ngpu > 1):
        net_G = nn.parallel(net_G, list(range(args.ngpu)))
    net_G.apply(weight_init)    # 应用权值
    # net_G.load_state_dict(torch.load('checkpoint/netG_epoch_10.pth', map_location=torch.device('cpu')))

    net_D = Discriminator_w(ngpu = args.ngpu, nc = nc, ndf = args.ndf).to(device) # 初始化判别器

    if (device.type == 'cuda') and (args.ngpu > 1):
        net_D = nn.parallel(net_D, list(range(args.ngpu)))
    net_D.apply(weight_init)    # 应用权值
    # net_D.load_state_dict(torch.load('checkpoint/netD_epoch_10.pth', map_location=torch.device('cpu')))

    fixed_noise = torch.rand(args.batchSize, args.nz, 1, 1, device = device) # z

    optimizer_D = optim.RMSprop(net_D.parameters(), lr = 0.00005)      # 模型优化器
    optimizer_G = optim.RMSprop(net_G.parameters(), lr = 0.00005)

    G_losses = []
    D_losses = []

    print("开始训练")  # 训练过程
    for epoch in range(args.epochs + 100):  # 训练循环次数
        net_D.train()
        net_G.train()
        for step, data in enumerate(dataloader, 0):
            # 更新D网络
            net_D.zero_grad()   # 梯度置零

            # Take all real batch
            real = data.to(device)
            b_size = real.size(0)
            output = net_D(real).view(-1)
            errD_real = - torch.mean(output)
            D_x = output.mean().item()

            # Take all fake batch
            noise = torch.randn(b_size, args.nz, 1, 1, device=device)
            fake = net_G(noise)
            output = net_D(fake.detach()).view(-1)
            errD_fake = torch.mean(output)
            D_G_z1 = output.mean().item()

            errD = errD_real + errD_fake

            errD.backward()
            optimizer_D.step()

            for p in net_D.parameters():
                p.data.clamp_(- 0.01, 0.01) # 数据裁剪

            if step % 5 == 0:
                # 更新G网络
                net_G.zero_grad()
                output = net_D(fake).view(-1)
                errG = - torch.mean(output)
                errG.backward()
                D_G_z2 = output.mean().item()
                optimizer_G.step()

                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, step, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            if step % 100 == 0:
                vutils.save_image(real, '%s/real_samples.png' % (args.outf),
                                  normalize = True)
                fake = net_G(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d.png' % (args.outf, epoch),
                                  normalize = True)

            G_losses.append(errG.item())
            D_losses.append(errD.item())
            print(G_losses)
            print(D_losses)
        if epoch % 5 == 0:
            torch.save(net_G.state_dict(), '%s/netG_epoch_%d.pth' % (args.ckpt, epoch))
            torch.save(net_D.state_dict(), '%s/netD_epoch_%d.pth' % (args.ckpt, epoch))

    plt.figure(figsize = (10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label = "G")
    plt.plot(D_losses, label = "D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    args = get_args()
    if args.model == "1":
        train(args)
    else:
        train_w(args)
