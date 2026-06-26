import warnings
warnings.filterwarnings('ignore')
import os
import argparse
import torch
from networks.net_factory import net_factory
from utils.test_patch import test_all_case,test_single_case_first_output
import h5py
import numpy as np
import pyvista as pv
import numpy as np
from skimage import measure
import math
from medpy import metric
import torch.nn.functional as F
from tqdm import tqdm

def test_all_case(model, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, dataset_name="LA"):
    if dataset_name == "LA":
        with open('../data/LA/test.list', 'r') as f:
            image_list = f.readlines()

        image_list = ["../data/LA/" + item.replace('\n', '') + "/mri_norm2.h5" for item in image_list]
    elif dataset_name == "Pancreas_CT":
        with open('../data/Pancreas/test.list', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/Pancreas/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]
    elif dataset_name == "BraTS2019":
        with open('../data/BraTS2019/test.txt', 'r') as f:
            image_list = f.readlines()
        image_list = ["../data/BraTS2019/BraTSh5/" + item.replace('\n', '') + ".h5" for item in image_list]
    loader = tqdm(image_list)
    total_dice = 0.0
    total_jc = 0.0
    total_hd = 0.0
    total_asd = 0.0
    for image_path in loader:
        h5f = h5py.File(image_path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        label[label<0] = 0
        prediction, score_map = test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=num_classes)
        if np.sum(prediction)==0:
            dice, jc, hd, asd = 0, 0, 0, 0
        else:
            dice = metric.binary.dc(prediction, label) # metric.binary.dc(prediction, label)
            jc = metric.binary.jc(prediction, label)
            hd = metric.binary.hd95(prediction, label)
            asd = metric.binary.asd(prediction, label)
        total_dice += dice
        total_jc += jc
        total_hd += hd
        total_asd += asd
    avg_dice = total_dice / len(image_list)
    avg_jc = total_jc / len(image_list)
    avg_hd = total_hd / len(image_list)
    avg_asd = total_asd / len(image_list)
    print(f'average dice: {avg_dice*100:.2f}, average jc: {avg_jc*100:.2f}, average hd: {avg_hd:.2f}, average asd: {avg_asd:.2f}')
    return avg_dice, avg_jc, avg_hd, avg_asd

def test_single_case_first_output(model, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0]-w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1]-h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2]-d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad//2,w_pad-w_pad//2
    hl_pad, hr_pad = h_pad//2,h_pad-h_pad//2
    dl_pad, dr_pad = d_pad//2,d_pad-d_pad//2
    if add_pad:
        image = np.pad(image, [(wl_pad,wr_pad),(hl_pad,hr_pad), (dl_pad, dr_pad)], mode='constant', constant_values=0)
    ww,hh,dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes, ) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy*x, ww-patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y,hh-patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd-patch_size[2])
                test_patch = image[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch,axis=0),axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y = model(test_patch)
                    if len(y) > 1:
                        y = y[0]
                    y = F.softmax(y, dim=1)
                y = y.cpu().data.numpy()
                y = y[0,1,:,:,:]
                score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = score_map[:, xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + y
                cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] \
                  = cnt[xs:xs+patch_size[0], ys:ys+patch_size[1], zs:zs+patch_size[2]] + 1

    score_map = score_map/np.expand_dims(cnt,axis=0)
    label_map = (score_map[0]>0.5).astype(np.int32)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
        score_map = score_map[:,wl_pad:wl_pad+w,hl_pad:hl_pad+h,dl_pad:dl_pad+d]
    return label_map, score_map



parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../', help='Name of Experiment')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--exp', type=str,  default='debug', help='exp_name')
parser.add_argument('--labelnum', type=int, default=8, help='labeled data')
parser.add_argument('--model', type=str,  default='vnet', help='model_name')
parser.add_argument('--alpha_max', type=float, default=1.0, help='maximum value of alpha')
parser.add_argument('--alpha_min', type=float, default=0.99, help='minimum value of alpha')
parser.add_argument('--savefreq', type=int, default=200, help='frequency of model saving (in epochs)')
parser.add_argument('--topk', type=int, default=16, help='topk for adamix')
parser.add_argument('--per_num', type=int, default=8, help='each dimension of the feature map is divided into how many patches for adamix')
parser.add_argument('--age', type=float, default=1., help='age for adamix')
parser.add_argument('--weight_psemix', type=float, default=0.5, help='weight for pseudo_mix loss')
parser.add_argument('--optim', type=str, default='adamw', help='optimizer for training(adamw or sgd)')
parser.add_argument('--adamw_lr', type=float, default=1e-4, help='learning rate for adamw')
parser.add_argument('--decay', type=float, default=1e-2, help='weight decay for adamw')
parser.add_argument('--iter', type=int, default=60000, help='test iteration')
parser.add_argument('--iter_test', type=int, default=0, help='whether to test the model saved at iter_test (if iter_test=0, then test the best model)')




FLAGS = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
if FLAGS.dataset_name == "BraTS2019" and FLAGS.iter_test == 1:
    model_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}/almax{}-almin{}-savefreq{}-topk{}-per_num{}-age{}-weight_psemix{}-{}-{}-{}/{}_{}_best_model.pth".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model, FLAGS.alpha_max, FLAGS.alpha_min, FLAGS.savefreq, FLAGS.topk, FLAGS.per_num, FLAGS.age, FLAGS.weight_psemix, FLAGS.optim, FLAGS.adamw_lr, FLAGS.decay, FLAGS.exp, FLAGS.iter)
else:
    model_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}/almax{}-almin{}-savefreq{}-topk{}-per_num{}-age{}-weight_psemix{}-{}-{}-{}/{}_best_model.pth".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model, FLAGS.alpha_max, FLAGS.alpha_min, FLAGS.savefreq, FLAGS.topk, FLAGS.per_num, FLAGS.age, FLAGS.weight_psemix, FLAGS.optim, FLAGS.adamw_lr, FLAGS.decay, FLAGS.exp)

test_save_path = FLAGS.root_path + "model/{}_{}_{}_labeled/{}_predictions/".format(FLAGS.dataset_name, FLAGS.exp, FLAGS.labelnum, FLAGS.model)

num_classes = 2

if FLAGS.dataset_name == "LA":
    patch_size = (112, 112, 80)
    stride_xy = 18
    stride_z = 4
    
elif FLAGS.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    stride_xy = 32
    stride_z = 32
    
elif FLAGS.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    stride_xy = 16
    stride_z = 16

if __name__ == '__main__':
        model = net_factory(net_type=FLAGS.model, in_chns=1, class_num=num_classes, mode='test').cuda()
        model.load_state_dict(torch.load(model_path), strict=False)
        model.eval()
        test_all_case(model=model, num_classes=num_classes, patch_size=patch_size, stride_xy=stride_xy, stride_z=stride_z, dataset_name=FLAGS.dataset_name)
# data = h5py.File(r'D:\pycode\SGRS-Net\data\LA\0RZDK210BSMWAA6467LU\mri_norm2.h5', 'r')
# image = data['image']
# predict, _ = test_single_case_first_output(model, image, stride_xy=18, stride_z=4, patch_size=(112,112,80), num_classes=2)
# print(predict.shape, np.unique(predict))
# verts, faces, _, _ = measure.marching_cubes(predict, 0.5)

# faces = np.hstack([[3, *f] for f in faces])
# mesh = pv.PolyData(verts, faces)
# # mesh = mesh.smooth(n_iter=100)
# plotter = pv.Plotter()
# plotter.add_mesh(mesh, color='red', smooth_shading=True)

# plotter.show()


