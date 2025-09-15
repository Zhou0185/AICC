##########################################################################
## Creator: Cihsiang
## Email: f09921058@ntu.edu.tw
##########################################################################
import os, sys
import shutil
from pathlib import Path
import argparse
import datetime
import random
import time

import cv2
import torch
from tensorboardX import SummaryWriter
import warnings
warnings.filterwarnings('ignore')
import pdb

# Custom Modules
from engine import *
#from datasets import build_dataset
from models import build_model
import util.misc as utils
from util.logger import setup_logger, AvgerageMeter, EvaluateMeter
from PIL import Image, ImageDraw
# In[0]: Parser
def parse_args():
    from config import cfg, merge_from_file, merge_from_list
    parser = argparse.ArgumentParser('APGCC')
    # parser.add_argument('-c', '--config_file', type=str, default="", help='the path to the training config')
    #parser.add_argument('-c', '--config_file', type=str, default="/home/hp/zrj/prjs/APGCC-main/apgcc/configs/AICC_IFI.yml", help='the path to the training config')
    parser.add_argument('-c', '--config_file', type=str, default="/home/hp/zrj/prjs/APGCC-main/apgcc/configs/AICC_test.yml", help='the path to the training config')


    parser.add_argument('-t', '--test', action='store_true', default=False, help='Model test')

    parser.add_argument('opts', help='overwriting the training config' 
                        'from commandline', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()


    if args.config_file != "":
        cfg = merge_from_file(cfg, args.config_file)
    cfg = merge_from_list(cfg, args.opts)
    cfg.config_file = args.config_file
    cfg.test = args.test
    return cfg

# In[1]: Main
def main():
    # Initial Config and environment
    cfg = parse_args()
    print("cfg.GPU_ID:",cfg.GPU_ID)
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(cfg.GPU_ID)
    device = torch.device('cuda')
    seed = cfg.SEED
    if seed != None:
        g = torch.Generator()
        g.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        os.environ['PYTHONHASHSEED'] = str(seed)

    # Function

        #test(cfg)
    source_dir = cfg.OUTPUT_DIR
    output_dir = os.path.join(source_dir, "%s_%.2f"%(cfg.DATASETS.DATASET, cfg.TEST.THRESHOLD))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    vis_val_path = None
    if cfg.VIS:
        vis_val_path = os.path.join(output_dir, 'sample_result_for_val/')
        if not os.path.exists(vis_val_path):
            os.makedirs(vis_val_path)

    # logging
    num_gpus = torch.cuda.device_count()
    logger = setup_logger('AGPCC', output_dir, 0)
    logger.info('Using {} GPUS'.format(num_gpus))
    logger.info('Running with config:\n{}'.format(cfg))
    logger.info("##############################################################")
    logger.info("# % TEST .....  ")
    logger.info("# Config is {%s}" %(cfg.config_file))
    logger.info("# SEED is {%s}" %(cfg.SEED))
    logger.info("# DATASET is {%s}" %(cfg.DATASETS.DATASET))
    logger.info("# DATA_RT is {%s}" %(cfg.DATASETS.DATA_ROOT))
    logger.info("# MODEL.ENCODER is {%s}" %(cfg.MODEL.ENCODER))
    logger.info("# MODEL.DECODER is {%s}" %(cfg.MODEL.DECODER))
    logger.info("# RESUME is {%s}" %(cfg.RESUME))
    logger.info("# BATCH is {%d*%d*%d}"%(cfg.SOLVER.BATCH_SIZE, cfg.MODEL.ROW, cfg.MODEL.LINE))
    logger.info("# OUTPUT_DIR is {%s}" %(cfg.OUTPUT_DIR))
    logger.info("# RESULT_DIR is {%s}" %(output_dir))
    logger.info("##############################################################")
    logger.info('Eval Log %s' % time.strftime("%c"))

    # Define the dataset.
    #train_dl, val_dl = build_dataset(cfg=cfg)
    torch.multiprocessing.set_sharing_strategy('file_system')  # avoid limitation of number of open files.

    # Building the Model & Optimizer
    model = build_model(cfg=cfg, training=False)
    model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('number of params:%d \n' % n_parameters)

    pretrained_dict = torch.load(os.path.join("/home/hp/zrj/prjs/pth", 'APGCC_NEFCell_best_e20.pth'))
    print(source_dir)
    # pretrained_dict = torch.load(os.path.join(source_dir, 'SHHA_best.pth'), map_location='cpu')
    # pretrained_dict = torch.load(os.path.join(source_dir, 'latest.pth'), map_location='cpu')


    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)

    def load_data(img_path, gt_path):
        ###################################
        # return imgs, points
        ###################################
        # load the images
        # print(img_path)
        img = cv2.imread(img_path)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # load ground truth points
        points = []
        with open(gt_path) as f_label:
            for line in f_label:
                x = float(line.strip().split(' ')[0])
                y = float(line.strip().split(' ')[1])
                points.append([x, y])
        return img, np.array(points)


    # Starting Testing
    print("Start testing")
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    f1s = []
    precisions = []
    recalls = []
    threshold = 0.5
    root_path = "/home/hp/zrj/prjs/APGCC-main/apgcc/datasets/CELLSsplit_v4_test/DATA_ROOT/test"
    img_name = "IMG_111.tif"
    img_path = root_path + "/images/" + img_name
    print(img_path)
    # 得到gt的点坐标，不只是count
    gt_path_root = root_path + "/test_file"
    gt_path = gt_path_root + "/" + img_name.split('.')[0] + ".txt"
    img = cv2.imread(img_path)
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    img = torch.nn.functional.upsample_bilinear(img.unsqueeze(0), scale_factor=1).squeeze(0)
    img = torch.Tensor(img).unsqueeze(0)#将{Tensor:(3,1024,1024)}变为{Tensor:(1,3,1024,1024)}
    samples = img
    samples = samples.to(device)  # print(samples.shape)
    # run inference

    outputs = model(samples)
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]

    outputs_points = outputs['pred_points'][0]


    with open(gt_path, 'r') as f:
        points = f.readlines()
        # print(points)
        p_gt = [[int(point.split(" ")[0]), int(point.split(" ")[1].strip())] for point in
                points]
        print("len(p_gt):", len(p_gt))
    #gt_cnt = targets[0]['point'].shape[0]
    gt_cnt = len(p_gt)
    # print("gt_cnt:",gt_cnt)
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    predict_cnt = int((outputs_scores > threshold).sum())

    # pdb.set_trace()
    threshold = 0.5
    # threshold = 10#无knn
    #TP = HungarianMatch(targets[0]['point'], points, threshold)
    TP = HungarianMatch(p_gt, points, threshold)

    anchornum = 4096
    F1point_score_ori, Precision_score_ori, Recall_score_ori = pointF1_score(TP, p_gt, points)

    # accumulate MAE, MSE
    mae = abs(predict_cnt - gt_cnt)
    mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
    print("predict_cnt:",predict_cnt)
    print("mae:",mae," mse",mse)
    # t1 = time.time()
    # result = evaluate_crowd_counting(model, val_dl, next(model.parameters()).device, cfg.TEST.THRESHOLD, vis_val_path)
    # t2 = time.time()
    # # logger.info('Eval: MAE=%.6f, MSE=%.6f time:%.2fs \n'%(result[0], result[1], t2 - t1))
    # logger.info('Eval: MAE=%.6f, MSE=%.6f, F1=%.6f, PRECISION=%.6f, RECALL=%.6f time:%.2fs \n'
    #             % (result[0], result[1], result[2], result[3], result[4], t2 - t1))



# %%
if __name__ == '__main__':
    main()
