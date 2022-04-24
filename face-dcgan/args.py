import argparse

def get_args(): # 参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='1', help='训练模型')
    parser.add_argument('--dataroot', type = str, default = './image', help = '存放数据的地址')
    parser.add_argument('--workers', type = int, default = 2, help = '进行数据预处理及数据加载使用进程数')
    parser.add_argument('--batchSize', type = int, default = 64, help = '一次batch进入模型的图片数目')
    parser.add_argument('--imageSize', type = int, default = 64, help = '原始图片重采样进入模型前的大小')
    parser.add_argument('--nz', type = int, default = 100, help = '初始噪音向量的大小')
    parser.add_argument('--ngf', type = int, default = 64, help = '生成网络中基础feature数目')
    parser.add_argument('--ndf', type = int, default = 64, help = '判别网络中基础feature数目')
    parser.add_argument('--epochs', type = int, default = 100, help = '训练过程中epoch数目')
    parser.add_argument('--lr', type = float, default = 2e-3, help = '初始学习率 默认0.002')
    parser.add_argument('--beta1', type = float, default = 0.5, help = '使用Adam优化算法中的bata1参数值')
    parser.add_argument('--cuda', action = 'store_true', help = '指定使用GPU进行训练')
    parser.add_argument('--ngpu', type = int, default = 1, help = '使用gpu数量')
    parser.add_argument('--netG', default = '', help = "指定生成网络的检查点文件")
    parser.add_argument('--netD', default = '', help = "指定判别网络的检查点文件")
    parser.add_argument('--outf', default = './fake-image', help = '模型输出图片保存路径')
    parser.add_argument('--ckpt', default = './checkpoint', help = '检查点保存路径')
    parser.add_argument('--manualSeed', type = int, help = '指定生成随机数的seed')

    args = parser.parse_args()
    return args