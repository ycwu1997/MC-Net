import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from utils import ramps, losses, metrics, test_patch
from dataloaders.dataset import *
from networks.net_factory import net_factory

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def sharpening(P):
    T = 1/args.temperature
    P_sharpen = P ** T / (P ** T + (1-P) ** T)
    return P_sharpen

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', type=str,  default='LA', help='dataset_name')
parser.add_argument('--root_path', type=str, default='./', help='Name of Dataset')
parser.add_argument('--exp', type=str,  default='MCNet', help='exp_name')
parser.add_argument('--model', type=str,  default='mcnet3d_v1', help='model_name')
parser.add_argument('--max_iteration', type=int,  default=15000, help='maximum iteration to train')
parser.add_argument('--max_samples', type=int,  default=80, help='maximum samples to train')
parser.add_argument('--labeled_bs', type=int, default=2, help='batch_size of labeled data per gpu')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size of labeled data per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--labelnum', type=int,  default=16, help='trained samples')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=1, help='consistency_weight')
parser.add_argument('--consistency_rampup', type=float, default=40.0, help='consistency_rampup')
parser.add_argument('--temperature', type=float, default=0.1, help='temperature of sharpening')
parser.add_argument('--lamda', type=float, default=0.5, help='weight to balance all losses')
args = parser.parse_args()

snapshot_path = args.root_path + "model/{}_{}_{}_labeled/{}".format(args.dataset_name, args.exp, args.labelnum, args.model)

num_classes = 2
if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = args.root_path+'data/LA'
    args.max_samples = 80
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = args.root_path+'data/Pancreas'
    args.max_samples = 62
train_data_path = args.root_path

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
labeled_bs = args.labeled_bs
max_iterations = args.max_iteration
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('./code/', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    
    model = net_factory(net_type=args.model, in_chns=1, class_num=num_classes, mode="train")
    if args.dataset_name == "LA":
        db_train = LAHeart(base_dir=train_data_path,
                        split='train',
                        transform = transforms.Compose([
                            RandomRotFlip(),
                            RandomCrop(patch_size),
                            ToTensor(),
                            ]))
    elif args.dataset_name == "Pancreas_CT":
        db_train = Pancreas(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    labelnum = args.labelnum  
    labeled_idxs = list(range(labelnum))
    unlabeled_idxs = list(range(labelnum, args.max_samples))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))
    consistency_criterion = losses.mse_loss
    dice_loss = losses.Binary_dice_loss
    iter_num = 0
    best_dice = 0
    max_epoch = max_iterations // len(trainloader) + 1
    lr_ = base_lr
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):
            
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            model.train()
            outputs = model(volume_batch)
            num_outputs = len(outputs)

            y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)

            loss_seg = 0
            loss_seg_dice = 0 
            for idx in range(num_outputs):
                y = outputs[idx][:labeled_bs,...]
                y_prob = F.softmax(y, dim=1)
                loss_seg += F.cross_entropy(y[:labeled_bs], label_batch[:labeled_bs])
                loss_seg_dice += dice_loss(y_prob[:,1,...], label_batch[:labeled_bs,...] == 1)

                y_all = outputs[idx]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)

            loss_consist = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_consist += consistency_criterion(y_ori[i], y_pseudo_label[j])
            
            iter_num = iter_num + 1
            consistency_weight = get_current_consistency_weight(iter_num//150)
            
            loss = args.lamda * loss_seg_dice + consistency_weight * loss_consist
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logging.info('iteration %d : loss : %03f, loss_d: %03f, loss_cosist: %03f' % (iter_num, loss, loss_seg_dice, loss_consist))
        
            writer.add_scalar('Labeled_loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('Labeled_loss/loss_seg_ce', loss_seg, iter_num)
            writer.add_scalar('Co_loss/consistency_loss', loss_consist, iter_num)
            writer.add_scalar('Co_loss/consist_weight', consistency_weight, iter_num)

            if iter_num >= 800 and iter_num % 200 == 0:
                ins_width = 2
                B,C,H,W,D = y_ori[0].size()
                snapshot_img = torch.zeros(size = (D, 3, (num_outputs + 2) *H + (num_outputs + 2) * ins_width, W + ins_width), dtype = torch.float32)

                target =  label_batch[labeled_bs,...].permute(2,0,1)
                train_img = volume_batch[labeled_bs,0,...].permute(2,0,1)

                snapshot_img[:, 0,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 1,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))
                snapshot_img[:, 2,:H,:W] = (train_img-torch.min(train_img))/(torch.max(train_img)-torch.min(train_img))

                snapshot_img[:, 0, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 1, H+ ins_width:2*H+ ins_width,:W] = target
                snapshot_img[:, 2, H+ ins_width:2*H+ ins_width,:W] = target

                snapshot_img[:,:, :,W:W+ins_width] = 1
                for idx in range(num_outputs+2):
                    begin_grid = idx+1
                    snapshot_img[:,:, begin_grid*H + ins_width:begin_grid*H + begin_grid*ins_width,:] = 1

                for idx in range(num_outputs):
                    begin = idx + 2
                    end = idx + 3
                    snapshot_img[:, 0, begin*H+ begin*ins_width:end*H+ begin*ins_width,:W] = y_ori[idx][labeled_bs:][0,1].permute(2,0,1)
                    snapshot_img[:, 1, begin*H+ begin*ins_width:end*H+ begin*ins_width,:W] = y_ori[idx][labeled_bs:][0,1].permute(2,0,1)
                    snapshot_img[:, 2, begin*H+ begin*ins_width:end*H+ begin*ins_width,:W] = y_ori[idx][labeled_bs:][0,1].permute(2,0,1)
                writer.add_images('Epoch_%d_Iter_%d_unlabel'% (epoch_num, iter_num), snapshot_img)
                
            if iter_num >= 800 and iter_num % 200 == 0:
                model.eval()
                if args.dataset_name =="LA":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=18, stride_z=4, dataset_name = 'LA')
                elif args.dataset_name =="Pancreas_CT":
                    dice_sample = test_patch.var_all_case(model, num_classes=num_classes, patch_size=patch_size, stride_xy=16, stride_z=16, dataset_name = 'Pancreas_CT')
                if dice_sample > best_dice:
                    best_dice = dice_sample
                    save_mode_path = os.path.join(snapshot_path,  'iter_{}_dice_{}.pth'.format(iter_num, best_dice))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    torch.save(model.state_dict(), save_mode_path)
                    torch.save(model.state_dict(), save_best_path)
                    logging.info("save best model to {}".format(save_mode_path))
                writer.add_scalar('Var_dice/Dice', dice_sample, iter_num)
                writer.add_scalar('Var_dice/Best_dice', best_dice, iter_num)
                model.train()

            if iter_num >= max_iterations:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
