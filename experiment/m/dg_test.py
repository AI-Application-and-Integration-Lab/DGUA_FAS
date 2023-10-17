import sys

sys.path.append('../../')
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional as F
from config import config
from util.utils import sample_frames
from util.dataset import YunpeiDataset
from util.utils import AverageMeter, accuracy, draw_roc
from util.statistic import get_EER_states, get_HTER_at_thr, calculate_threshold
from sklearn.metrics import roc_auc_score
from option import get_training_arguments
from cvnets import get_model





os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus

accs = [[], [], []]
accs2 = [[], [], []]
def test(test_dataloader, model, threshold):
    valid_top1 = AverageMeter()
    prob_dict = {}
    label_dict = {}
    p_dict = {}
    model.eval()
    output_dict_tmp = {}
    target_dict_tmp = {}
    number = 0

    tmp = []
    na = [] 
    test_features_dict = {}
    with torch.no_grad():
        for iter, (input, target, videoID) in enumerate(test_dataloader):
            if (iter % 100 == 0):
                    print('**Testing** ', iter, ' photos done!')
            input = Variable(input).cuda()
            target = Variable(torch.from_numpy(np.array(target)).long()).cuda()
            cls_out= model(input, config.norm_flag)[0]
            prob = F.softmax(cls_out, dim=1).cpu().data.numpy()[:, 1]
            p = F.softmax(cls_out, dim=1).detach().cpu().numpy()
           
            tmp += [[p, torch.argmax(cls_out, dim=1)[0].item(), target.detach().cpu()[0].item()]]
            # novel attack
            if tmp[-1][2] == 3:
                na += [prob] 
            label = target.cpu().data.numpy() 
            videoID = videoID.cpu().data.numpy()
            

            for i in range(len(prob)):
                if (videoID[i] in prob_dict.keys()):
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    p_dict[videoID[i]].append(p[i])
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 3))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                else:
                    prob_dict[videoID[i]] = []
                    label_dict[videoID[i]] = []
                    p_dict[videoID[i]] = []
                    prob_dict[videoID[i]].append(prob[i])
                    label_dict[videoID[i]].append(label[i])
                    p_dict[videoID[i]].append(p[i])
                    output_dict_tmp[videoID[i]] = []
                    target_dict_tmp[videoID[i]] = []
                    output_dict_tmp[videoID[i]].append(cls_out[i].view(1, 3))
                    target_dict_tmp[videoID[i]].append(target[i].view(1))
                    # 1*3*256*256
                    input = input[i].reshape(1, input[i].shape[0], input[i].shape[1], input[i].shape[2])
                    # feature: 640*8*8; flatten: 40960
                    feature = torch.flatten(model.extract_features(input.cuda())).detach().cpu()
                    test_features_dict[videoID[i]] = feature.numpy()
                    number += 1
        
    print('**Testing** ', number, ' photos done!')
    
    prob_list = []
    label_list = []
    p_list = []
    test_features = []
    for key in prob_dict.keys():
        avg_single_video_prob = sum(prob_dict[key]) / len(prob_dict[key])
        avg_single_video_label = sum(label_dict[key]) / len(label_dict[key])
        avg_single_video_p = sum(p_dict[key]) / len(p_dict[key])
        test_features.append(test_features_dict[key])
        prob_list = np.append(prob_list, avg_single_video_prob)
        label_list = np.append(label_list, avg_single_video_label)
        p_list.append(avg_single_video_p) 
        # compute loss and acc for every video
        avg_single_video_output = sum(output_dict_tmp[key]) / len(output_dict_tmp[key])
        avg_single_video_target = sum(target_dict_tmp[key]) / len(target_dict_tmp[key])
        acc_valid = accuracy(avg_single_video_output, torch.where(avg_single_video_target == 1, 1, 0), topk=(1,))
        valid_top1.update(acc_valid[0])

    binary_label = np.where(label_list == 1, np.ones_like(label_list), np.zeros_like(label_list))
    cur_EER_valid, threshold, FRR_list, FAR_list = get_EER_states(prob_list, binary_label)
    ACC_threshold = calculate_threshold(prob_list, binary_label, threshold)
    auc_score = roc_auc_score(binary_label, prob_list)
    draw_roc(FRR_list, FAR_list, auc_score)
    cur_HTER_valid = get_HTER_at_thr(prob_list, binary_label, threshold)
    

    return [valid_top1.avg, cur_EER_valid, cur_HTER_valid, auc_score, ACC_threshold, threshold]

def main():
    
    opts = get_training_arguments(config_path='./../../configs/mobilevit_s.yaml')
    net = get_model(opts)
    net.cuda()
    net_ = torch.load(config.best_model_path + config.tgt_best_model_name)
    net.load_state_dict(net_["state_dict"])
    threshold = net_["threshold"]
    net.eval()

    
    test_data = sample_frames(flag=2, num_frames=config.tgt_test_num_frames, dataset_name=config.tgt_data)
    test_dataloader = DataLoader(YunpeiDataset(test_data, train=False), batch_size=1, shuffle=False)
  
            
    
    print('\n')
    print("**Testing** Get test files done!")
    # test model
    test_args = test(test_dataloader, net, threshold)
    print('\n===========Test Info===========\n')
    print(config.tgt_data, 'Test acc: %5.4f' %(test_args[0]))
    print(config.tgt_data, 'Test EER: %5.4f' %(test_args[1]))
    print(config.tgt_data, 'Test HTER: %5.4f' %(test_args[2]))
    print(config.tgt_data, 'Test AUC: %5.4f' % (test_args[3]))
    print(config.tgt_data, 'Test ACC_threshold: %5.4f' % (test_args[4]))
    print('\n===============================\n')


    
if __name__ == '__main__':
    main()
