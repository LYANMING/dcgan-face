from torch.utils.data import Dataset
import PIL.Image as Image
import os

class FaceDataset(Dataset):     # 数据处理，对图片进行处理,用来批量加载数据
    def __init__(self, root, transform = None): # 数据集初始化
        super(FaceDataset, self).__init__() # 调用父类初始化方法
        self.imgs = [root + '/' + i for i in os.listdir(root)]  # 存储图片列表
        self.transform = transform  # 类型转换
        print("Number of data:", len(self.imgs))    # 打印图片列表长度

    def __getitem__(self, index):   # 支持从 0 到 len(self)的索引
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):  # 数据集的大小
        return len(self.imgs)