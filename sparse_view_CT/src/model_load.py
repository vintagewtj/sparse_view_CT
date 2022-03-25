import pylab
import torch

from dataset_generator import *
from model import *
device = torch.device("cuda:0")

i = 4                        # 指定使用训练第几个epoch后所保存的模型进行测试
# h5_seq = 0                   # 指定使用第几个h5中的文件进行测试
# tensor_seq = 0               # 指定使用该h5文件中第几个数据进行测试
j = 4                          # 指定使用训练集中第几个数据进行测试
test_data = Datasets('../original_data/LDCT_DATA/ground_truth_test')
save_path = "../outputs/model{0}_outputs{1}.bmp".format(i, j)

test_img = test_data[j][0][0]

vol_geom = astra.create_vol_geom(362, 362)
proj_geom = astra.create_proj_geom('parallel', 1.0, 513,
                                    np.linspace(np.pi * -90 / 180, np.pi * 90 / 180, 60, False))
P = np.array(test_img)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
sinogram_id, sinogram = astra.create_sino(P, proj_id)
rec_id = astra.data2d.create('-vol', vol_geom)
cfg = astra.astra_dict('FBP_CUDA')  # FBP_CUDA
cfg['ReconstructionDataId'] = rec_id
cfg['ProjectionDataId'] = sinogram_id
# cfg['option'] = {'FilterType': 'Ram-lak'}
cfg['ProjectorId'] = proj_id

alg_id = astra.algorithm.create(cfg)
astra.algorithm.run(alg_id)
astra.algorithm.run(alg_id, 20)
test_img = astra.data2d.get(rec_id)
transforms = torchvision.transforms.ToTensor()
test_img = transforms(test_img)

test_img = test_img.view(1, 1, 362, 362)
model = torch.load("../checkpoints/model_{}.pth".format(i), )
model.to(device)
with torch.no_grad():
    test_img = test_img.to(device)
    output = model(test_img)

print(output)
output = output.cpu()
out_img = np.array(output)
out_img = np.reshape(out_img, (362, 362))
pylab.gray()
pylab.figure(1)
pylab.imshow(out_img)
pylab.show()
pylab.imsave(save_path, out_img)

