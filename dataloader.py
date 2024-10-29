import os
import random
import shutil
import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F


def rm_mkdir(dir_path):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
        print('Remove path - %s'%dir_path)
    os.makedirs(dir_path)
    print('Create path - %s'%dir_path)


class make_dataset(Dataset):
    def __init__(self,img_root,gt_root,mode='train',augmentation_prob=0.4):
        super(make_dataset,self).__init__()
        self.img_root = img_root
        self.gt_root = gt_root
        self.img_paths = os.listdir(self.img_root)
        self.img_paths = [os.path.join(self.img_root,x) for x in self.img_paths]
        self.img_paths.sort(key=lambda x : int(x.split('\\')[-1][:-len('.jpg')]))
        self.gt_paths = os.listdir(self.gt_root)
        self.gt_paths = [os.path.join(self.gt_root,x) for x in self.gt_paths]
        self.gt_paths.sort(key=lambda x : int(x.split('\\')[-1][:-len('.jpg')]))
        self.mode = mode
        self.augmentation_prob = augmentation_prob
        self.RotationDegree = [0,90,180,270]

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        gt_path = self.gt_paths[idx]
        img = Image.open(img_path)
        gt = Image.open(gt_path)
        Transform = []
        p_transform = random.random()
        if (self.mode=='train') and (p_transform <= self.augmentation_prob):
            RotationDegree = random.randint(0,3)
            RotationDegree = self.RotationDegree[RotationDegree]
            Transform.append(T.RandomRotation((RotationDegree,RotationDegree)))
            RotationDegree_2 = random.randint(-10,10)
            Transform.append(T.RandomRotation((RotationDegree_2,RotationDegree_2)))
            Transform.append(T.RandomCrop(256))
            Transform =T.Compose(Transform)
            img = Transform(img)
            gt = Transform(gt)

            if random.random() < 0.5:
                img = F.hflip(img)
                gt = F.hflip(gt)

            if random.random() < 0.5:
                img = F.vflip(img)
                gt = F.vflip(gt)
            Transform = []
        Transform.append(T.CenterCrop(256))
        Transform.append(T.ToTensor())
        Transform = T.Compose(Transform)
        img = Transform(img)
        gt = Transform(gt)

        return img,gt

    def __len__(self):
        return len(self.img_paths)


# 将B/L的1000张图片分别分为训练集和测试集
# 将数据随机分为训练集和测试集
def mk_data_dir(img_path,target_path,p=0.7,stype='B'):
    # 获取每一组图片与标签的地址，并排序，以便通过索引找到图片
    img_path_list = [os.path.join(img_path,x) for x in os.listdir(img_path)]
    img_path_list.sort(key=lambda x: int(x.split('\\')[-1][:-len('.jpg')]))
    target_path_list = [os.path.join(target_path,x) for x in os.listdir(target_path)]
    target_path_list.sort(key=lambda x: int(x.split('\\')[-1][:-len('.jpg')]))
    train_len = int(len(img_path_list) * p)

    # 生成一个0-999的列表，随机分为两个列表，代表训练集和测试集的图片标号
    original_list = list(range(len(img_path_list)))
    sampled_list = random.sample(original_list, train_len)
    remaining_list = [item for item in original_list if item not in sampled_list]

    # 设置保存路径
    train_save_path = f'Dataset/train/train_{stype}'
    rm_mkdir(os.path.join(train_save_path,'img'))
    rm_mkdir(os.path.join(train_save_path,'target'))
    test_save_path = f'Dataset/test/test_{stype}'
    rm_mkdir(os.path.join(test_save_path, 'img'))
    rm_mkdir(os.path.join(test_save_path, 'target'))

    for i in sampled_list:
        name = f'{i + 1}.jpg'
        img = img_path_list[i]
        img = Image.open(img)
        img.save(os.path.join(train_save_path,'img',name))

        target = target_path_list[i]
        target = Image.open(target)
        target.save(os.path.join(train_save_path,'target',name))

    for i in remaining_list:
        name = f'{i + 1}.jpg'
        img = img_path_list[i]
        img = Image.open(img)
        img.save(os.path.join(test_save_path,'img',name))

        target = target_path_list[i]
        target = Image.open(target)
        target.save(os.path.join(test_save_path,'target',name))


# 获取数据加载器
def get_loader(img_root_1,gt_root_1,img_root_2,gt_root_2,mode='train',augmentation_prob=0.4,num_workers=0):
    if mode == 'train':
        dataset_B = make_dataset(img_root=img_root_1,
                                 gt_root=gt_root_1,
                                 mode=mode,
                                 augmentation_prob=augmentation_prob)
        dataset_L = make_dataset(img_root=img_root_2,
                                 gt_root=gt_root_2,
                                 mode=mode,
                                 augmentation_prob=augmentation_prob)
        train_dataset = dataset_B + dataset_L
        train_dataloader = DataLoader(dataset=train_dataset,
                                      shuffle=True,
                                      batch_size=8,
                                      num_workers=num_workers)
        return train_dataloader
    if mode == "test":
        dataset_B = make_dataset(img_root=img_root_1,
                                 gt_root=gt_root_1,
                                 mode=mode,
                                 augmentation_prob=augmentation_prob)
        dataset_L = make_dataset(img_root=img_root_2,
                                 gt_root=gt_root_2,
                                 mode=mode,
                                 augmentation_prob=augmentation_prob)
        test_dataset = dataset_B + dataset_L
        test_dataloader = DataLoader(dataset=test_dataset,
                                     shuffle=True,
                                     batch_size=8,
                                     num_workers=num_workers)
        return test_dataloader


if __name__ == '__main__':

    img_path_B = r'Dataset/B/inputs'
    target_path_B = r'Dataset/B/targets'
    mk_data_dir(img_path=img_path_B,target_path=target_path_B,stype='B')

    img_path_L = r'Dataset/L/inputs'
    target_path_L = r'Dataset/L/targets'
    mk_data_dir(img_path=img_path_L,target_path=target_path_L,stype='L')

    train_dataloader = get_loader(img_root_1=r'Dataset/train/train_B/img',
                                  gt_root_1=r'Dataset/train/train_B/target',
                                  img_root_2=r'Dataset/train/train_L/img',
                                  gt_root_2=r'Dataset/train/train_L/target',
                                  mode='train')
    test_dataloader = get_loader(img_root_1=r'Dataset/test/test_B/img',
                                 gt_root_1=r'Dataset/test/test_B/target',
                                 img_root_2=r'Dataset/test/test_L/img',
                                 gt_root_2=r'Dataset/test/test_L/target',
                                 mode='test')
    # 测试加载器是否可以正常使用
    # for (img,gt) in test_dataloader:
    #     print(torch.max(img.view(-1).cpu()))
    #     output = img.view(-1).cpu()>= 0.5
    #     print(torch.max(output))
    #     target = gt.view(-1).cpu()
    #     print(torch.max(target))
    #     print(target)
    #     correct = (output == target).sum().item()
    #     total = output.size(0)
    #     acc = correct / total
    #     print(acc)
    #     break
    for (img,gt) in train_dataloader:
        print(gt.size())
        break