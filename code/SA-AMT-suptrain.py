# Adaptive Mean Teacher with Structure-Aware Block Mixing for Semi-Supervised Medical Image Segmentation

import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import warnings
import torch.nn.functional as F
warnings.filterwarnings('ignore')
import argparse
import logging
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss
import torch.optim as optim
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from utils import ramps, losses,  test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.mixmatch_util import mix_match_just_k1, mix_match_cross_select_pseudolabel2
from datetime import datetime
from utils.util2 import WeightAdEMA
from utils.adamix_utils import AdaMix3D_boud

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str, default='LA', help='Pancreas_CT,LA,BraTS2019')
parser.add_argument('--root_path', type=str, default='../', help='Name of Dataset')
parser.add_argument('--exp', type=str, default='debug', help='exp_name')
parser.add_argument('--model', type=str, default='vnet', help='model_name')
parser.add_argument('--max_iteration', type=int, default=30000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int, default=80, help='maximum samples to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size of labeled data per gpu')  # 2
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--savefreq', type=int, default=200, help='frequency of model saving (in epochs)')
parser.add_argument('--adamw_lr', type=float, default=1e-4, help='learning rate for adamw')
parser.add_argument('--decay', type=float, default=1e-2, help='weight decay for adamw')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}/savefreq{}-adamw-{}-{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model, args.savefreq, args.adamw_lr, args.decay)

num_classes = 2
if args.dataset_name == "LA":
    # patch_size = (32, 32, 32)     #debug
    patch_size = (112, 112, 80)
    # args.root_path = args.root_path + 'data/LA'
    train_data_path = '../data/LA'
    args.max_samples = 80
    eval_start = 6000
    
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    # patch_size = (64, 64, 64)     #debug
    # args.root_path = args.root_path + 'data/Pancreas'
    train_data_path = '../data/Pancreas'
    args.max_samples = 62
    eval_start = 6000
    
elif args.dataset_name == "BraTS2019":
    patch_size = (96, 96, 96)
    # patch_size = (48, 48, 48) #debug
    # args.root_path = args.root_path + 'data/BraTS2019'
    train_data_path = '../data/BraTS2019'
    # args.model = 'vnets'
    args.max_samples = 250
    args.max_iteration = 60000
    eval_start = 26000
# train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
max_iterations = args.max_iteration


if __name__ == "__main__":
    
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(args)
#     logging.info("%s alpha=%s", args, alpha)


    def create_model(ema=False):
        # Network definition
        net = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model


    model = create_model(ema=False)

    if args.dataset_name == "LA":
        db_train = LAHeart_no_read(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                        #        RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),with_idx=True)
    elif args.dataset_name == "Pancreas_CT":   # Pancreas_no_read # Pancreas
        db_train = Pancreas_no_read(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]),with_idx=True)
    elif args.dataset_name == "BraTS2019":
        db_train = BraTS2019_no_read(base_dir=train_data_path,
                             split='train',
                             transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                             ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = Subset(db_train, labeled_idxs)
    trainloader = DataLoader(db_train, batch_size=args.batch_size, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.adamw_lr, betas=(0.9, 0.999), weight_decay=args.decay)
    
    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    # dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    max_epoch = max_iterations // len(trainloader) + 1
    iter_per_epoch = len(trainloader)
    
    iterator = tqdm(range(max_epoch), ncols=70)

    start = datetime.now()
    time_start = start.strftime("%Y-%m-%d %H:%M:%S")
    print('start-time', time_start)

    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            l_image = volume_batch#.cpu().numpy()
            l_label = label_batch#.cpu().numpy()

        
            model.train()

            
            X_data = l_image.cuda()
            X_label = l_label.cuda().float()
            
            
            X = X_data
            output_all= model(X)
            output_all_softmax = torch.softmax(output_all, dim=1)
            
            
            loss_seg_ce_lab = ce_loss(output_all, X_label.long())
            loss_seg_dice_lab = dice_loss(output_all_softmax, X_label.long().unsqueeze(1))
            loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)
            
           
            
            iter_num = iter_num + 1
            
        
            writer.add_scalar('loss', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if iter_num >= eval_start and iter_num % args.savefreq == 0:                
                now = datetime.now()
                time_str = now.strftime("%Y-%m-%d %H:%M:%S")
                print(time_str)
                
                model.eval()

                if args.dataset_name =="LA":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas_CT":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=32, stride_z=32, dataset_name = 'Pancreas_CT')
                elif args.dataset_name =="BraTS2019":
                    dice_sample, jc_sample, hd_sample, asd_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'BraTS2019')

                if dice_sample > best_dice:
                    best_dice = dice_sample
                    if args.dataset_name =="BraTS2019":
                        save_mode_path = os.path.join(snapshot_path,  f'{args.exp}_{iter_num}_best_model.pth')
                    else:
                        save_mode_path = os.path.join(snapshot_path,  f'{args.exp}_best_model.pth')
                    torch.save(model.state_dict(), save_mode_path)
                    logging.info(f"save best model to {save_mode_path}")
                logging.info(f"iter {iter_num}, dice_sample: [{dice_sample*100:.2f}/{best_dice*100:.2f}]")
                
                
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                
                model.train()
            
            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                print('best_dice',best_dice)
                break
            
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

