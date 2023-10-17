import sys
sys.path.append('../../')

from util.utils import save_checkpoint, AverageMeter, Logger, accuracy, mkdirs, time_to_str
from util.evaluate import eval
from util.get_loader import get_dataset

import random
import numpy as np
from config import config
from datetime import datetime
import time
from timeit import default_timer as timer
import os
import torch
import torch.nn as nn
import torch.optim as optim 
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter

from cvnets.models import get_model
from option import get_training_arguments

class SmoothCrossEntropy(nn.Module):
    def __init__(self, alpha=0.5):
        super(SmoothCrossEntropy, self).__init__()
        self.alpha = alpha

    def forward(self, logits, labels):
        num_classes = logits.shape[-1]
        alpha_div_k = self.alpha / num_classes
        target_probs = nn.functional.one_hot(labels, num_classes=num_classes).float() * \
            (1. - self.alpha) + alpha_div_k
        loss = -(target_probs * torch.log_softmax(logits, dim=-1)).sum(dim=-1)
        return loss.mean()
        

random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
torch.cuda.manual_seed(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
device = 'cuda'

def train():
    mkdirs(config.checkpoint_path, config.best_model_path, config.logs)
    # load data
    src1_train_dataloader_fake, src1_train_dataloader_real, \
    src2_train_dataloader_fake, src2_train_dataloader_real, \
    src3_train_dataloader_fake, src3_train_dataloader_real, \
    tgt_valid_dataloader = get_dataset(config.src1_data, config.src1_train_num_frames, 
                                       config.src2_data, config.src2_train_num_frames, 
                                       config.src3_data, config.src3_train_num_frames,
                                       config.tgt_data, config.tgt_test_num_frames, config.batch_size)

    best_model_ACC = 0.0
    best_model_HTER = 1.0
    best_model_ACER = 1.0
    best_model_AUC = 0.0
    # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:ACER, 5:AUC, 6:threshold
    valid_args = [np.inf, 0, 0, 0, 0, 0, 0, 0]

    loss_classifier = AverageMeter()
    classifer_top1 = AverageMeter()

    opts = get_training_arguments(config_path='./../../configs/mobilevit_s.yaml')
    net = get_model(opts).to(device)
    net2 = get_model(opts).to(device)
    
    state_dict = torch.load('./../../pretrained_model/mobilevit_s.pt')
    del state_dict['classifier.fc.weight']
    del state_dict['classifier.fc.bias']
    net.load_state_dict(state_dict, strict=False)
    net2.load_state_dict(state_dict, strict=False)

    writer = SummaryWriter('./logs/runs')
    log = Logger()
    log.open(config.logs + config.tgt_data + '_log.txt', mode='a')
    log.write("\n----------------------------------------------- [START %s] %s\n\n" % (
    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '-' * 51))

    log.write('** start training target model! **\n')
    log.write(
        '--------|------------- VALID -------------|--- classifier ---|------ Current Best ------|--------------|\n')
    log.write(
        '  iter  |   loss   top-1   HTER    AUC    |   loss   top-1   |   top-1   HTER    AUC    |    time      |\n')
    log.write(
        '-------------------------------------------------------------------------------------------------------|\n')
    start = timer()
    criterion = {
        'softmax': nn.CrossEntropyLoss(label_smoothing=0.1).cuda(),
        'l1': nn.L1Loss().cuda(),
        'lsr_hard' : SmoothCrossEntropy(0.5),
        'lsr_easy' : SmoothCrossEntropy(1.0)
    }
    optimizer_dict = [
        {"params": filter(lambda p: p.requires_grad, net.parameters()), "lr": config.init_lr},
    ]
    optimizer_dict2 = [
        {"params": filter(lambda p: p.requires_grad, net2.parameters()), "lr": config.init_lr},
    ]

    optimizer = optim.Adam(optimizer_dict, lr=config.init_lr, weight_decay=config.weight_decay)
    optimizer2 = optim.Adam(optimizer_dict2, lr=config.init_lr, weight_decay=config.weight_decay)
    init_param_lr = []
    for param_group in optimizer.param_groups:
        init_param_lr.append(param_group["lr"])

    iter_per_epoch = 10

    src1_train_iter_real = iter(src1_train_dataloader_real)
    src1_iter_per_epoch_real = len(src1_train_iter_real)
    src2_train_iter_real = iter(src2_train_dataloader_real)
    src2_iter_per_epoch_real = len(src2_train_iter_real)
    src3_train_iter_real = iter(src3_train_dataloader_real)
    src3_iter_per_epoch_real = len(src3_train_iter_real)

    src1_train_iter_fake = iter(src1_train_dataloader_fake)
    src1_iter_per_epoch_fake = len(src1_train_iter_fake)
    src2_train_iter_fake = iter(src2_train_dataloader_fake)
    src2_iter_per_epoch_fake = len(src2_train_iter_fake)
    src3_train_iter_fake = iter(src3_train_dataloader_fake)
    src3_iter_per_epoch_fake = len(src3_train_iter_fake)
    
    max_iter = config.max_iter
    epoch = 1
    if(len(config.gpus) > 1):
        net = torch.nn.DataParallel(net).cuda()
        net2 = torch.nn.DataParallel(net).cuda()

    for iter_num in range(max_iter+1):
        if (iter_num % src1_iter_per_epoch_real == 0):
            src1_train_iter_real = iter(src1_train_dataloader_real)
        if (iter_num % src2_iter_per_epoch_real == 0):
            src2_train_iter_real = iter(src2_train_dataloader_real)
        if (iter_num % src3_iter_per_epoch_real == 0):
            src3_train_iter_real = iter(src3_train_dataloader_real)
       
        if (iter_num % src1_iter_per_epoch_fake == 0):
            src1_train_iter_fake = iter(src1_train_dataloader_fake)
        if (iter_num % src2_iter_per_epoch_fake == 0):
            src2_train_iter_fake = iter(src2_train_dataloader_fake)
        if (iter_num % src3_iter_per_epoch_fake == 0):
            src3_train_iter_fake = iter(src3_train_dataloader_fake)
        
        if (iter_num != 0 and iter_num % iter_per_epoch == 0):
            epoch = epoch + 1
        param_lr_tmp = []
        for param_group in optimizer.param_groups:
            param_lr_tmp.append(param_group["lr"])

        net.train(True)
        optimizer.zero_grad()
      
        ######### data prepare #########
        src1_img_real, src1_label_real = src1_train_iter_real.next()
        src1_img_real = src1_img_real.cuda()
        src1_label_real = src1_label_real.cuda()

        src2_img_real, src2_label_real = src2_train_iter_real.next()
        src2_img_real = src2_img_real.cuda()
        src2_label_real = src2_label_real.cuda()
        
        src3_img_real, src3_label_real = src3_train_iter_real.next()
        src3_img_real = src3_img_real.cuda()
        src3_label_real = src3_label_real.cuda()    

        src1_img_fake, src1_label_fake = src1_train_iter_fake.next()
        src1_img_fake = src1_img_fake.cuda()
        src1_label_fake = src1_label_fake.cuda()
       

        src2_img_fake, src2_label_fake = src2_train_iter_fake.next()
        src2_img_fake = src2_img_fake.cuda()
        src2_label_fake = src2_label_fake.cuda()
      

        src3_img_fake, src3_label_fake = src3_train_iter_fake.next()
        src3_img_fake = src3_img_fake.cuda()
        src3_label_fake = src3_label_fake.cuda()
       

        input_data = torch.cat([src1_img_real, src1_img_fake, src2_img_real, src2_img_fake, src3_img_real, src3_img_fake], dim=0)

        source_label = torch.cat([src1_label_real, src1_label_fake,
                                  src2_label_real, src2_label_fake,
                                  src3_label_real, src3_label_fake,
                                 ], dim=0)

        ######### forward #########
        ######### Copycat train #########
        bsz = source_label.size(0)
        net.train(False)
        net2.train(True) # Copycat Model
        optimizer2.zero_grad()
        classifier_label_out, x11, x12, x13 = net(input_data, return_feature=True)
        classifier_label_out2, x21, x22, x23 = net2(input_data, return_feature=True)
        
        pullloss1 = criterion["l1"](x11.reshape(bsz, -1),x21.reshape(bsz,-1))
        pullloss2 = criterion["l1"](x12.reshape(bsz, -1),x22.reshape(bsz,-1))
        cls_loss = criterion["softmax"](classifier_label_out2.narrow(0, 0, input_data.size(0)), source_label)
        
        pullloss = (pullloss1 + pullloss2) / 2
        
        cls_loss = cls_loss + pullloss
        cls_loss.backward()
        optimizer2.step()


        ######## MainModel train ########
        net.train(True)
        net2.train(False) # Copycat Model
        optimizer.zero_grad()
        classifier_label_out, x11, x12, x13 = net(input_data, return_feature=True)
        classifier_label_out2, x21, x22, x23 = net2(input_data, return_feature=True)
        
        
        out21 = net(input_data, x1 = x21)
        out22 = net(input_data, x2 = x22)
        out23 = net(input_data, x3 = x23)
       
        klu0 = criterion["lsr_hard"](out21, source_label)
        klu1 = criterion["lsr_hard"](out22, source_label)
        klu2 = criterion["lsr_easy"](out23, source_label)
        klu = (klu0 + klu1 + klu2) / 3

        # features_dim = 20*640*8*8
        real_features = net.extract_features(input_data[source_label == 1])
        
        l1_loss = criterion["l1"](real_features, torch.zeros_like(real_features))

        ######### cross-entropy loss #########
        cls_loss = criterion["softmax"](classifier_label_out.narrow(0, 0, input_data.size(0)), source_label)

        ######### backward #########
        total_loss = cls_loss + l1_loss + 0.1 * klu
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss_classifier.update(cls_loss.item())
        acc = accuracy(classifier_label_out.narrow(0, 0, input_data.size(0)), source_label, topk=(1,))
        classifer_top1.update(acc[0])
        print('\r', end='', flush=True)
        print(
            '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s'
            % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'))
            , end='', flush=True)

        if (iter_num != 0 and (iter_num+1) % iter_per_epoch == 0):
            train_loss = loss_classifier.avg
            train_acc = classifer_top1.avg
            # 0:loss, 1:top-1, 2:EER, 3:HTER, 4:AUC, 5:threshold, 6:ACC_threshold
            valid_args = eval(tgt_valid_dataloader, net)
            # judge model according to HTER
            is_best = valid_args[4] > best_model_AUC
            best_model_AUC = max(valid_args[4], best_model_AUC)
            threshold = valid_args[5]
            if is_best:
                best_model_ACC = valid_args[1]
                best_model_HTER = valid_args[3]

            save_list = [epoch, valid_args, best_model_HTER, best_model_ACC, best_model_ACER, threshold]
            save_checkpoint(save_list, is_best, net, config.gpus, config.checkpoint_path, config.best_model_path)
            print('\r', end='', flush=True)
            log.write(
                '  %4.1f  |  %5.3f  %6.3f  %6.3f  %6.3f  |  %6.3f  %6.3f  |  %6.3f  %6.3f  %6.3f  | %s   %s'
                % (
                (iter_num+1) / iter_per_epoch,
                valid_args[0], valid_args[1], valid_args[3] * 100, valid_args[4] * 100,
                loss_classifier.avg, classifer_top1.avg,
                float(best_model_ACC), float(best_model_HTER * 100), float(best_model_AUC * 100),
                time_to_str(timer() - start, 'min'),
                param_lr_tmp[0]))
            log.write('\n')
            writer.add_scalars(f'Accuracy', {'train': train_acc[0].detach(),
                                             'valid': valid_args[1][0].detach()}, (iter_num + 1) // iter_per_epoch)
            writer.add_scalars(f'Loss', {'train': train_loss,
                                         'valid': valid_args[0]}, (iter_num + 1) // iter_per_epoch)
            writer.add_scalar(f'AUC', valid_args[4], (iter_num + 1) // iter_per_epoch)
            writer.add_scalar(f'HTER', valid_args[3], (iter_num + 1) // iter_per_epoch)
            time.sleep(0.01)

if __name__ == '__main__':
    train()


