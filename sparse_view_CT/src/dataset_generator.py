import os
import h5py
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import astra
import pylab


class Datasets(Dataset):
    def __init__(self, path):
        super(Datasets, self).__init__()
        self.root = path
        self.h5_list = os.listdir(self.root)
        self.dataset = None

    def __getitem__(self, idx):
        if self.dataset is None:
            self.h5_seq = idx // 128
            self.h5_dir = os.path.join(self.root, self.h5_list[self.h5_seq])
            self.tensor_seq = idx % 128
            targets = h5py.File(self.h5_dir, 'r')['data'][self.tensor_seq]
            vol_geom = astra.create_vol_geom(362, 362)
            proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                               np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 60, False))
            P = np.array(targets)
            proj_id = astra.create_projector('cuda', proj_geom, vol_geom)  # CPU类型的的投影生成模式可以是
                                                                         # line, linear, strip, GPU的话使用cuda即可。
            sinogram_id, sinogram = astra.create_sino(P, proj_id)
            rec_id = astra.data2d.create('-vol', vol_geom)
            cfg = astra.astra_dict('FBP_CUDA')  # 进行网络训练的同时使用GPU重建占用大量资源，所以不使用FBP_CUDA
            cfg['ReconstructionDataId'] = rec_id
            cfg['ProjectionDataId'] = sinogram_id
            cfg['option'] = {'FilterType': 'Ram-lak'}  # 如果使用SIRT等迭代算法，此行舍去
            cfg['ProjectorId'] = proj_id
            alg_id = astra.algorithm.create(cfg)
            astra.algorithm.run(alg_id)
            #astra.algorithm.run(alg_id, 20)          # 如果使用SIRT等迭代算法，此行指定迭代次数

            imgs = astra.data2d.get(rec_id)

            transform = transforms.ToTensor()
            imgs = transform(imgs)
            imgs = imgs.view(1, 362, 362)

            targets = transform(targets)
            targets = targets.view(1, 362, 362)

            astra.algorithm.delete(alg_id)
            astra.data2d.delete(rec_id)
            astra.data2d.delete(sinogram_id)
            astra.projector.delete(proj_id)

        return imgs, targets


    def __len__(self):
        return len(self.h5_list) * 128

if __name__ == '__main__':
    train_data = Datasets('../original_data/LDCT_DATA/ground_truth_train')
    test_data = Datasets('../original_data/LDCT_DATA/ground_truth_test')
    a = test_data[4][0]
    b = test_data[4][1]
    a = a.view(362, 362)
    b = b.view(362, 362)
    pylab.gray()
    pylab.figure(1)
    pylab.imshow(a)
    pylab.figure(2)
    pylab.imshow(b)
    pylab.show()
    # for i in range(len(train_data)):
    #     train_input = train_data[i][0]
    #     train_input = train_input.view(362, 362)
    #     train_input_path = "../dataset/train/ground_truth/input{}.bmp".format(i)
    #     pylab.save(train_input_path, train_input)
    #
    #     train_target = train_data[i][1]
    #     train_target = train_input.view(362, 362)
    #     train_input_path = "../dataset/train/targets/target{}.bmp".format(i)
    #     pylab.save(train_input_path, train_target)
