# 鉴于原始图片较大，为了更好地训练，将图片统一裁剪为288*288像素大小
from PIL import Image
import os
import random


def crop(proto_imgs_path,proto_labels_path,proto_noiseimgs_path,name):
    # 获取要裁剪的图片与对应特征点位置与标签图片的地址
    imgs_path = proto_imgs_path
    labels_path = proto_labels_path
    noiseimgs_path = proto_noiseimgs_path

    print(f'原始图片路径：{imgs_path}')
    print(f'原始标签路径：{labels_path}')
    print(f'原始噪声图片路径：{noiseimgs_path}')

    # 获取每一张图片的地址并排序
    imgs_list = [os.path.join(imgs_path,path) for path in os.listdir(imgs_path)]
    imgs_list.sort(key= lambda x :int(x.split('\\')[-1][:-4]))

    labels_list = [os.path.join(labels_path,path) for path in os.listdir(labels_path)]
    labels_list.sort(key= lambda x :int(x.split('\\')[-1][:-4]))

    noiseimgs_list = [os.path.join(noiseimgs_path, path) for path in os.listdir(noiseimgs_path)]
    noiseimgs_list.sort(key=lambda x: int(x.split('\\')[-1][:-4]))

    # 设置裁剪后图片的保存路径并创建文件夹
    saveimgs_path = f'CroppedDataset/{name}/imgs'
    savelabels_path = f'CroppedDataset/{name}/labels'
    savenoiseimgs_path = f'CroppedDataset/{name}/noise_imgs'

    print(f'裁剪后图片路径：{saveimgs_path}')
    print(f'裁剪后标签路径：{savelabels_path}')
    print(f'裁剪后噪声图片路径：{savenoiseimgs_path}')

    if not os.path.exists(saveimgs_path):
        os.makedirs(saveimgs_path)
    if not os.path.exists(savelabels_path):
        os.makedirs(savelabels_path)
    if not os.path.exists(savenoiseimgs_path):
        os.makedirs(savenoiseimgs_path)

    if len(imgs_list) == len(labels_list) == len(noiseimgs_list):
        num = len(imgs_list)
        print(f'{name}数据集长度：{num}')
    else:
        print('imgs 和labels和 noiseimgs 数量不匹配')
        return None

    # 逐一裁剪图片并保存新图片和新label
    for i in range(num):
        # 读取图片
        image = Image.open(imgs_list[i])
        noiseimg = Image.open(noiseimgs_list[i])
        

        with open(labels_list[i]) as f:

            # 读取标签内容
            text = f.readline().split(' ')
            ori_x = float(text[0])
            ori_y = float(text[1])
            ori_w = float(text[2])
            ori_h = float(text[3])

            # 随机偏移中心一定距离
            x_skewing = random.uniform(-5., 5.)
            y_skewing = random.uniform(-5., 5.)
            x_cut = int(ori_x + x_skewing)
            y_cut = int(ori_y + y_skewing)
            left = x_cut - 144
            right = x_cut + 144
            top = y_cut - 144
            bottom = y_cut + 144

            # 裁剪图片
            image = image.crop((left, top, right, bottom))
            noiseimg = noiseimg.crop((left, top, right, bottom))

            # 保留旧名称并保存图片
            new_name = imgs_list[i].split('\\')[-1]
            new_lname = labels_list[i].split('\\')[-1]
            new_nname = noiseimgs_list[i].split('\\')[-1]
            image.save(os.path.join(saveimgs_path,new_name))
            noiseimg.save(os.path.join(savenoiseimgs_path,new_nname))

            # 计算新的label并保存
            new_x = ori_x - left
            new_y = ori_y - top
            new_w = 144
            new_h = 144
            txt_file = os.path.join(savelabels_path, new_lname)
            with open(txt_file, mode='a') as txt_file:
                print(new_x, new_y, new_w, new_h, file=txt_file)

    print("————————图片裁剪完成————————\n")


if __name__ == "__main__":
    # 对B类图片进行裁剪
    crop(proto_imgs_path=r'ProtoDataset/scan/B/imgs',
         proto_labels_path=r'ProtoDataset/scan/B/labels',
         proto_noiseimgs_path=r'ProtoDataset/scan/B/noise_imgs',
         name='B')
    # 对L类图片进行裁剪
    crop(proto_imgs_path=r'ProtoDataset/scan/L/imgs',
         proto_labels_path=r'ProtoDataset/scan/L/labels',
         proto_noiseimgs_path=r'ProtoDataset/scan/L/noise_imgs',
         name='L')