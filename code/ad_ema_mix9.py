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
from torch.utils.data import DataLoader
from utils import ramps, losses,  test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory
from utils.mixmatch_util import mix_match_just_k1, mix_match_cross_select_pseudolabel2, mix_match_cross_fuse_pseudolabel
from datetime import datetime
from utils.util2 import WeightEMAo, WeightAdEMA, WeightEMA
from utils.adamix_utils import AdaMix3D, AdaMix3D2, AdaMix3D_boud

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
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')  # 2
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')  # 4
parser.add_argument('--base_lr', type=float, default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int, default=8, help='trained samples')
parser.add_argument('--seed', type=int, default=1337, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--alpha', type=float, default=0.99, help='alpha start value')
parser.add_argument('--alpha_max', type=float, default=1.0, help='maximum value of alpha')
parser.add_argument('--alpha_min', type=float, default=0.99, help='minimum value of alpha')
parser.add_argument('--savefreq', type=int, default=200, help='frequency of model saving (in epochs)')
parser.add_argument('--topk', type=int, default=16, help='topk for adamix')
parser.add_argument('--per_num', type=int, default=8, help='each dimension of the feature map is divided into how many patches for adamix')
parser.add_argument('--age', type=float, default=0., help='age for adamix')
parser.add_argument('--weight_psemix', type=float, default=0.5, help='weight for pseudo_mix loss')
parser.add_argument('--optim', type=str, default='adamw', help='optimizer for training(adamw or sgd)')
parser.add_argument('--adamw_lr', type=float, default=1e-4, help='learning rate for adamw')
parser.add_argument('--decay', type=float, default=1e-2, help='weight decay for adamw')


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}/almax{}-almin{}-savefreq{}-topk{}-per_num{}-age{}-weight_psemix{}-{}-{}-{}".format(args.dataset_name, args.exp, args.labelnum,
                                                                    args.model,args.alpha_max, args.alpha_min, args.savefreq, args.topk, args.per_num, args.age, args.weight_psemix, args.optim, args.adamw_lr, args.decay)

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
    args.max_samples = 250
    args.max_iteration = 60000
    eval_start = 26000
# train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr



def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def compute_alpha_from_softdice(student_logits,
                                teacher_logits,
                                alpha_min=0.99,
                                alpha_max=0.999,
                                eps=1e-6):

    ps = F.softmax(student_logits, dim=1)
    pt = F.softmax(teacher_logits, dim=1)

    # remove background
    ps = ps[:, 1:]
    pt = pt[:, 1:]

    B, C = ps.shape[:2]

    ps = ps.reshape(B, C, -1)
    pt = pt.reshape(B, C, -1)

    intersection = (ps * pt).sum(dim=2)

    denominator = (
        (ps * ps).sum(dim=2) +
        (pt * pt).sum(dim=2)
    )

    soft_dice = (2 * intersection + eps) / (denominator + eps)

    # mean over batch & class
    soft_dice = soft_dice.mean()

    # difference
    dice_diff = 1.0 - soft_dice

    # adaptive EMA
    alpha = alpha_min + (alpha_max - alpha_min) * dice_diff

    alpha = alpha.clamp(alpha_min, alpha_max)

    return soft_dice.detach(), alpha.detach()

# from utils.util2 import FullAugmentor
# augmentor = FullAugmentor()

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
    ema_model = create_model(ema=True)
    if args.dataset_name == "LA":
        db_train = LAHeart_no_read(base_dir=train_data_path,
                           split='train',
                           transform=transforms.Compose([
                               RandomRotFlip(),
                               RandomCrop(patch_size),
                               ToTensor(),
                           ]),with_idx=True)
    elif args.dataset_name == "Pancreas_CT":   # Pancreas_no_read # Pancreas
        db_train = Pancreas_no_read(base_dir=train_data_path,
                            split='train',
                            transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomCrop(patch_size),
                                ToTensor(),
                            ]),with_idx=True)
    elif args.dataset_name == "BraTS2019":
        db_train = BraTS2019_no_read(base_dir=train_data_path,
                             split='train',
                             transform=transforms.Compose([
                                RandomRotFlip(),
                                RandomRot(),
                                RandomGenerator(),
                                RandomCrop(patch_size),
                                ToTensor(),
                             ]))
    labelnum = args.labelnum
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    
    if args.optim == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.adamw_lr, betas=(0.9, 0.999), weight_decay=args.decay)
    if args.optim == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    
    ema_optimizer = WeightAdEMA(model, ema_model)
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
    
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    augu = torch.tensor(1).cuda()
    start = datetime.now()
    time_start = start.strftime("%Y-%m-%d %H:%M:%S")
    print('start-time', time_start)
    adamix = AdaMix3D_boud(total_steps=max_iterations // iter_per_epoch, num_classes=num_classes, image_size=patch_size, patch_size=args.per_num, topk=args.topk, age=args.age, mode='hard', p=0.5)

    for epoch_num in iterator:
        
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            l_image = volume_batch[:args.labeled_bs]#.cpu().numpy()
            l_label = label_batch[:args.labeled_bs]#.cpu().numpy()

            ul_image = volume_batch[args.labeled_bs:]
            X = list(zip(l_image, l_label))
            
            
            X_prime, Umix, Lmix, pseudo_label, tl_label, X_cap, t_u, t_l = mix_match_cross_select_pseudolabel2(X,ul_image, eval_net=ema_model, K=1, alpha=0.75,
                                                          mixup_mode='_x', aug_factor=augu)
        
            model.train()

            
            X_data = l_image.cuda()
            X_label = l_label.cuda().float()
            
        #     U_data_m = Umix.cuda()
        #     U_mix_label = Lmix.cuda().float()
            
            U_data = ul_image.cuda()
            U_data_pseudo = pseudo_label.cuda().float()

            tl_label = tl_label.cuda().float()
            
            
            X = torch.cat((X_data,U_data), 0)
            output_all= model(X)
            output_all_softmax = torch.softmax(output_all, dim=1)
            conf_l = output_all_softmax[:args.labeled_bs].max(dim=1)[0]
            conf_u = output_all_softmax[args.labeled_bs:args.labeled_bs*2].max(dim=1)[0]
            
            U_data_m, U_mix_label, pred_l_conf, adloss, tok_list = adamix(oimage=U_data, aimage=X_data, olabel=pseudo_label.long(), alabel=tl_label.long(), oconf=conf_u, aconf=conf_l, prediction=output_all[args.labeled_bs:args.labeled_bs*2], cur_step= epoch_num)

            U_data_m = U_data_m.cuda()
            U_mix_label = U_mix_label.cuda().float()

            output_m = model(U_data_m)
            output_m_softmax = torch.softmax(output_m, dim=1)
            
            loss_seg_ce_lab = ce_loss(output_all[:args.labeled_bs], X_label.long())
            loss_seg_dice_lab = dice_loss(output_all_softmax[:args.labeled_bs], X_label.long().unsqueeze(1))
            supervised_loss = 0.5 * (loss_seg_ce_lab + loss_seg_dice_lab)
            
            # ce part
            output_all_l = output_all[:args.labeled_bs]
            output_all_u = output_all[args.labeled_bs:args.labeled_bs*2]
            output_all_u_mix = output_m
            # dice
            out_soft_u = output_all_softmax[args.labeled_bs:2*args.labeled_bs]
            out_soft_u_mix = output_m_softmax

            softdice, ad_alpha = compute_alpha_from_softdice(output_all_u, t_u, alpha_min=args.alpha_min, alpha_max=args.alpha_max, eps=1e-6)
        

            u_loss = 0.5 *(ce_loss(output_all_u, U_data_pseudo.long()) + dice_loss(out_soft_u, U_data_pseudo.long().unsqueeze(1)))
            umix_loss = 0.5 *(ce_loss(output_all_u_mix, U_mix_label.long()) + dice_loss(out_soft_u_mix, U_mix_label.long().unsqueeze(1)))


            pseudo_loss = u_loss * (1-args.weight_psemix) + umix_loss * args.weight_psemix
            
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            loss = supervised_loss + pseudo_loss 
        
        #     writer.add_scalar('ad_alpha', ad_alpha, iter_num)
        #     writer.add_scalar('jsd', jsd, iter_num)
            
            iter_num = iter_num + 1
            
            writer.add_scalar('ad_alpha', ad_alpha, iter_num)
            writer.add_scalar('softdice', softdice, iter_num)
            writer.add_scalars('adloss', {'adloss1':adloss[0], 'adloss2':adloss[1]}, iter_num)
            writer.add_scalars('topk', {'topk1':tok_list[0], 'topk2':tok_list[1]}, iter_num)
            writer.add_scalar('sup_loss', supervised_loss, iter_num)
            writer.add_scalar('u_loss', u_loss, iter_num)
            writer.add_scalar('umix_loss', umix_loss, iter_num)
            writer.add_scalar('loss', loss, iter_num)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_consist = 0

            ema_optimizer.step(alpha=ad_alpha)
            
            consistency_loss1 = 0
            # if iter_num % 100 == 0:
            #     logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (
            #         iter_num, loss, supervised_loss, loss_consist))
            
            
            
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
