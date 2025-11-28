import argparse
import math
import os
import pdb
import random
import re
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
import torchvision.transforms as standard_transforms
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from torch import nn

##import models_apgcc
##from models_apgcc import build_model
from sklearn.cluster import KMeans

from methods.apgcc.util.logger import setup_logger
from models import build_model
# pdb.set_trace()
del_re_time = []
add_time = []
# 添加新的配置参数来控制SSIM模式
SSIM_MODE = "rgb"  # 可设置为 "rgb" 或 "avg"或 "gray"
def DrawfPoints(points, Img_path):
        data = points
        llen = len(data)
        img = cv2.imread(Img_path)
        # noname = Img_path.split("/")[-1]
        noname = os.path.basename(Img_path).split(".")[0]
        dot_size = int(10 / 4096 * img.shape[0]) + 2
        for i in range(llen):
            # print(data[i],type(data[i]))
            x = int(data[i][0])
            y = int(data[i][1])
            tempmask = [x, y]
            # print(tempmask)
            img = cv2.circle(img, tempmask, dot_size, (255, 255, 255), -1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        plt.title(noname + " pd:" + str(llen))
        #plt.show()  # 显示图片
        return img
# def p2p_init_visual_counter():
#     parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
#     args = parser.parse_args()
#     print(args)
#     device = torch.device('cuda:{}'.format(args.gpu_id))
#     # get the P2PNet
#     #model = build_model(args)
#     model = build_model(cfg=args, training=False)
#
#     # move to GPU
#     model.to(device)
#     # load trained model
#     if args.weight_path is not None:
#         checkpoint = torch.load(args.weight_path, map_location='cpu')
#         model.load_state_dict(checkpoint['model'])
#         # convert to eval mode
#         model.eval()
#     # create the pre-processing transform
#     transform = standard_transforms.Compose([
#         standard_transforms.ToTensor(),
#         standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
#     return model,device,transform,args



def get_apgcc_args_parser():
    """
    为加载 APGCC 模型设置参数解析器。
    参照了 APGCC 项目 main.py 中的参数定义。
    """
    parser = argparse.ArgumentParser('Set parameters for APGCC model evaluation', add_help=False)

    # --- 模型相关参数 ---
    # 指定模型的骨干网络，默认为 'vgg16_bn'，与 APGCC 兼容
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="Name of the convolutional backbone to use")

    # 指定预训练权重的路径。注意：APGCC 习惯使用 --resume 作为参数名
    parser.add_argument('--resume', default='/home/hp/zrj/prjs/pth/APGCC_BCD_best.pth',
                        help='Path to the trained APGCC weights checkpoint')

    # 定义 P2PNet 风格的锚点行列数，这些参数在 APGCC 的 build_model 中也会被使用
    parser.add_argument('--row', default=2, type=int,
                        help="Row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="Line number of anchor points")

    # --- 设备相关参数 ---
    # 指定用于推理的 GPU ID
    parser.add_argument('--gpu_id', default=0, type=int,
                        help='The GPU used for evaluation')

    #
    parser.add_argument('--ssim_t_start', default=0.8, type=float,
                        help="start value for ssim threshold range")
    parser.add_argument('--ssim_t_end', default=0.8, type=float,
                        help="end value for ssim threshold range")
    parser.add_argument('--ssim_t_step', default=0.1, type=float,
                        help="step size for ssim threshold range")

    parser.add_argument('--t_view_start', default=0.1, type=float,
                        help="Start value for t_view range")
    parser.add_argument('--t_view_end', default=1, type=float,
                        help="End value for t_view range")
    parser.add_argument('--t_view_step', default=0.1, type=float,
                        help="Step size for t_view range")

    parser.add_argument('--t_candidate_start', default=0.01, type=float,
                        help="Start value for t_candidate range")
    parser.add_argument('--t_candidate_end', default=0.25, type=float,
                        help="End value for t_candidate range")
    parser.add_argument('--t_candidate_step', default=0.04, type=float,
                        help="Step size for t_candidate range")
    # 添加新的强度阈值参数
    parser.add_argument('--t_intensity_start', default=5, type=float,
                        help="Start value for t_candidate range")
    parser.add_argument('--t_intensity_end', default=40, type=float,
                        help="End value for t_candidate range")
    parser.add_argument('--t_intensity_step', default=5, type=float,
                        help="Step size for t_candidate range")
    return parser


import argparse
import torch
import torchvision.transforms as standard_transforms
from types import SimpleNamespace # 导入 SimpleNamespace，用于轻松创建嵌套对象

# 导入你重命名后的 APGCC 模型构建函数
# from models_apgcc import build_model

def get_apgcc_args_parser():
    """
    为加载 APGCC 模型设置参数解析器。
    这个函数只定义参数，并返回 parser 对象。
    """
    from config import cfg
    parser = argparse.ArgumentParser('APGCC_simulation')

    # --- APGCC 特有的配置参数 ---
    # APGCC 主要通过一个 .yml 配置文件来管理参数
    parser.add_argument('-c', '--config_file', type=str,
                        default="/home/hp/zrj/prjs/APGCC-main/apgcc/configs/AICC_BCDdata_test.yml",
                        help='The path to the APGCC config file')

    # 允许从命令行覆盖配置文件中的参数
    parser.add_argument('opts', help='Overwriting the training config from commandline',
                        default=None, nargs=argparse.REMAINDER)

    # --- 你的模拟脚本需要的自定义参数 ---
    parser.add_argument('--ssim_t_start', default=0.8, type=float,
                        help="start value for ssim threshold range")
    parser.add_argument('--ssim_t_end', default=0.8, type=float,
                        help="end value for ssim threshold range")
    parser.add_argument('--ssim_t_step', default=0.1, type=float,
                        help="step size for ssim threshold range")

    parser.add_argument('--t_view_start', default=0.5, type=float,
                        help="Start value for t_view range")
    parser.add_argument('--t_view_end', default=0.5, type=float,
                        help="End value for t_view range")
    parser.add_argument('--t_view_step', default=0.1, type=float,
                        help="Step size for t_view range")

    parser.add_argument('--t_candidate_start', default=0.05, type=float,
                        help="Start value for t_candidate range")
    parser.add_argument('--t_candidate_end', default=0.05, type=float,
                        help="End value for t_candidate range")
    parser.add_argument('--t_candidate_step', default=0.04, type=float,
                        help="Step size for t_candidate range")

    parser.add_argument('--t_intensity_start', default=20, type=float,
                        help="Start value for t_intensity range")
    parser.add_argument('--t_intensity_end', default=20, type=float,
                        help="End value for t_intensity range")
    parser.add_argument('--t_intensity_step', default=5, type=float,
                        help="Step size for t_intensity range")

    return parser


def apgcc_init_visual_counter():
    """
    初始化 APGCC 模型用于可视化计数。
    """
    # 导入 APGCC 的配置管理工具
    from config import cfg, merge_from_file, merge_from_list

    # 1. 获取定义好的解析器
    parser = get_apgcc_args_parser()

    # 2. 执行解析，这是整个脚本中唯一一次调用 parse_args
    args = parser.parse_args()

    # 3. 加载 APGCC 的配置文件 (这是 APGCC 的标准流程)
    if args.config_file != "":
        cfg = merge_from_file(cfg, args.config_file)
    if args.opts is not None:
        cfg = merge_from_list(cfg, args.opts)

    # 4. 设置设备
    device = torch.device('cuda:{}'.format(cfg.GPU_ID))
    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(cfg.GPU_ID)

    # (可选) 设置随机种子以保证可复现性
    seed = cfg.SEED
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 5. 构建模型架构 (APGCC 的 build_model 需要 cfg 对象)
    # 确保你已经正确导入了 APGCC 的 build_model: from models import build_model
    model = build_model(cfg=cfg, training=False)
    model.to(device)

    # 6. 加载预训练权重 (APGCC 的方式)
    pretrained_dict = torch.load(os.path.join("/home/hp/zrj/prjs/pth", 'APGCC_BCD_best.pth'), map_location='cpu')
    # pretrained_dict = torch.load(os.path.join(source_dir, 'SHHA_best.pth'), map_location='cpu')
    model_dict = model.state_dict()
    param_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(param_dict)
    model.load_state_dict(model_dict)
    # 7. 切换到评估模式
    model.eval()

    # 8. 创建图像预处理 transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 9. 返回与 p2p_init_visual_counter 一致的对象元组
    return model, device, transform, args
def ssim_rgb(y_true, y_pred):
    """
    分通道计算SSIM后取平均值（RGB模式）
    适用于彩色图像处理
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 如果是灰度图像（2维），则直接计算 SSIM
    if y_true.ndim == 2 and y_pred.ndim == 2:
        return calculate_ssim(y_true, y_pred)

    # 如果是彩色图像（3维），则分通道计算 SSIM
    if y_true.shape[2] != y_pred.shape[2]:
        raise ValueError("输入图像的通道数不匹配")

    ssim_channels = []
    for i in range(y_true.shape[2]):
        channel_true = y_true[:, :, i]
        channel_pred = y_pred[:, :, i]

        ssim = calculate_ssim(channel_true, channel_pred)
        ssim_channels.append(ssim)

    ssim_channels.remove(max(ssim_channels))
    return np.mean(ssim_channels)


def ssim_avg(y_true, y_pred):
    """
    先转换为灰度图再计算SSIM（AVG模式）
    更关注图像整体结构相似性
    """
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # 如果是灰度图像（2维），则直接计算 SSIM
    if y_true.ndim == 2 and y_pred.ndim == 2:
        return calculate_ssim(y_true, y_pred)

    # 如果是彩色图像（3维），则分通道计算 SSIM
    if y_true.shape[2] != y_pred.shape[2]:
        raise ValueError("输入图像的通道数不匹配")

    ssim_channels = []
    for i in range(y_true.shape[2]):
        channel_true = y_true[:, :, i]
        channel_pred = y_pred[:, :, i]

        ssim = calculate_ssim(channel_true, channel_pred)
        ssim_channels.append(ssim)

    # 计算所有通道的平均 SSIM
    return np.mean(ssim_channels)


def ssim_gray(y_true, y_pred):
    """
    将图像转换为灰度图像后计算SSIM
    不依赖外部库的实现
    """
    # 确保输入是numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 检测数据范围
    max_pixel = max(y_true.max(), y_pred.max())
    min_pixel = min(y_true.min(), y_pred.min())
    data_range = max_pixel - min_pixel

    # 如果数据范围很小或为0，使用默认范围255
    if data_range < 1.0:
        data_range = 255.0

    # 转换为灰度图像
    if y_true.ndim == 3 and y_true.shape[2] == 3:  # RGB图像
        y_true = (0.299 * y_true[:, :, 0] + 0.587 * y_true[:, :, 1] + 0.114 * y_true[:, :, 2])
    elif y_true.ndim == 3:  # 多通道非RGB
        y_true = np.mean(y_true, axis=2)

    if y_pred.ndim == 3 and y_pred.shape[2] == 3:  # RGB图像
        y_pred = (0.299 * y_pred[:, :, 0] + 0.587 * y_pred[:, :, 1] + 0.114 * y_pred[:, :, 2])
    elif y_pred.ndim == 3:  # 多通道非RGB
        y_pred = np.mean(y_pred, axis=2)

    # 确保是2D灰度图像
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError("灰度转换失败，结果是三维数组")

    # 计算灰度图像的SSIM
    return calculate_ssim(y_true, y_pred)
# 统一入口函数（根据配置选择不同模式）
def ssim(y_true, y_pred):
    """主SSIM计算函数，根据全局配置选择算法"""
    if SSIM_MODE == "rgb":
        return ssim_rgb(y_true, y_pred)
    elif SSIM_MODE == "avg":
        return ssim_avg(y_true, y_pred)
    elif SSIM_MODE == "gray":
        return ssim_gray(y_true, y_pred)
    else:
        raise ValueError(f"无效的SSIM模式: {SSIM_MODE}. 请使用 'rgb' 或 'avg'或 'gray'")

def calculate_ssim(y_true, y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)

    R = 255  # 假设像素值范围为[0, 255]
    c1 = np.square(0.01 * R)
    c2 = np.square(0.03 * R)

    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom

def interactive_adaptation_boxs_add(img_n, EXEMPLAR_BBX, Image_path, points_005,scores_005, current_points,current_scores,
                                    Display_width, Display_height, Image_Ori_W, Image_Ori_H, ssim_t=0.5, t_intensity=20):

    start = time.time()
    def resize_points(points):

        return [
            (round(int(point[0]) * Display_width // Image_Ori_W), round(int(point[1]) * Display_height // Image_Ori_H))
            for point in points
        ]

    def draw_point(points, index, img):

        draw = ImageDraw.Draw(img)
        draw.ellipse((points[index][0] - 2, points[index][1] - 2, points[index][0] + 2, points[index][1] + 2),
                     width=12, outline='red', fill=(255, 0, 0))
        return img

    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min
    crop_area_base = (x_min, y_min, x_max, y_max)
    # input_image = Image.open(Image_path).convert('RGB').resize((Display_width, Display_height), Image.LANCZOS)
    # input_image = Image.open(Image_path).resize((Display_width, Display_height))

    # Calculate bounding box dimensions
    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min

    input_image = Image.open(Image_path).resize((Display_width, Display_height))
    # plt.imshow(input_image)
    # plt.imsave("input_image.png", input_image)
    #plt.show()  # 显示图片
    # 检查裁剪区域是否有效
    # print(crop_area_base)
    # x_min, y_min, x_max, y_max = crop_area_base
    # if (x_min >= 0 and y_min >= 0 and
    #         x_max <= input_image.width and y_max <= input_image.height):
    #     print("裁剪区域有效")
    # else:
    #     print("裁剪区域超出图像范围")
    crop_img_base_rgb = input_image.crop(crop_area_base)

    # crop_img_array = np.array(crop_img_base_rgb)
    # print("裁剪图像像素值范围:", crop_img_array.min(), crop_img_array.max())
    #
    # # 显示裁剪图像
    # plt.imshow(crop_img_base_rgb, vmin=0, vmax=255)  # 假设像素值范围是 [0, 255]
    # plt.imsave("crop_img_base_rgb.png", crop_img_base_rgb)
    # plt.show()
    #rgb

    crop_img_base_rgb_array = np.array(crop_img_base_rgb)
    pixels_rgb = crop_img_base_rgb_array.reshape(-1, 3)
    #unique_points = np.unique(pixels_rgb, axis=0)
    #print("Unique points:", unique_points)
    #nn_clusters = len(unique_points)
    #print("nn_clusters",nn_clusters)
    #kmeans_rgb = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(pixels_rgb)
    kmeans_rgb = KMeans(n_clusters=2, random_state=0).fit(pixels_rgb)
    # kmeans_rgb = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(pixels_rgb)
    labels_rgb = kmeans_rgb.labels_
    labels_rgb_image = labels_rgb.reshape(crop_img_base_rgb_array.shape[:2])
    center_area_size = 5  # 5x5
    half_size = center_area_size // 2
    center_x, center_y = crop_img_base_rgb_array.shape[0] // 2, crop_img_base_rgb_array.shape[1] // 2
    center_area_labels = labels_rgb_image[
                         center_x - half_size:center_x + half_size + 1,
                         center_y - half_size:center_y + half_size + 1
                         ]
    cell_label = np.bincount(center_area_labels.ravel()).argmax()
    mask = (labels_rgb_image == cell_label).astype(np.uint8)
    cell_pixels = crop_img_base_rgb_array[mask == 1]
    #background_pixels = crop_img_base_rgb_array[mask == 0]
    average_cell_pixel = np.mean(cell_pixels.mean(axis=0))
    #average_background_pixel = background_pixels.mean(axis=0)
    #average_pixel_diff = np.mean(np.abs(average_cell_pixel - average_background_pixel))

    # Resize points
    resized_points = resize_points(points_005)
    points_raw = torch.tensor(resized_points).view(-1, 2).tolist()
    points_raw_copy = points_raw.copy()
    current_points_copy = current_points[:]
    current_scores_copy = current_scores[:]
    # img_raw = Image.open(Image_path).convert('L').resize((Display_width, Display_height), Image.LANCZOS)
    img_raw = Image.open(Image_path).convert('RGB').resize((Display_width, Display_height))
    img_array = np.array(img_raw)
    selected_points = []
    # start = time.time()
    for i, point in enumerate(points_raw):
        if point in selected_points:
            continue

        point[0] = min(point[0], Display_width - 1)
        point[1] = min(point[1], Display_height - 1)
        current_pixel = np.mean(img_array[points_raw[i][1]][points_raw[i][0]])  # ---注意翻一下横纵坐标
        #current_pixel = img_array[point[1]][point[0]]
        # if current_pixel < max_pixel // 2:
        #     continue
        #if (abs(current_pixel - average_cell_pixel) > 8):
        if (abs(current_pixel - average_cell_pixel) > t_intensity):
            # print("点像素与细胞像素差距过大。point's pixel likes background,continue ", "该点的像素值:", current_pixel,
            #       " 目标参考像素值average_cell_pixel：", average_cell_pixel)
            continue
        # Count points within the bounding box centered around the current point
        # pnum = sum(
        #     1 for pt in current_points_copy
        #     if (point[0] - x_len // 2 < int(pt[0]) < point[0] + x_len // 2 and
        #         point[1] - y_len // 2 < int(pt[1]) < point[1] + y_len // 2)
        # )
        pnum = sum(
            1 for pt in current_points_copy
            if (point[0] - x_len // 2 < int(pt[0]) < point[0] + x_len // 2 and
                point[1] - y_len // 2 < int(pt[1]) < point[1] + y_len // 2) and pt not in selected_points
        )

        if pnum == 0:
            # Define the crop area and perform SSIM comparison
            n_x_min, n_x_max = point[0] - x_len // 2, point[0] + x_len // 2
            n_y_min, n_y_max = point[1] - y_len // 2, point[1] + y_len // 2
            crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
            #crop_img = input_image.crop(crop_area).convert('L')
            crop_img = input_image.copy().crop(crop_area).convert('RGB')
            if crop_img.size != crop_img_base_rgb.size:
                crop_img = crop_img.resize(crop_img_base_rgb.size, Image.LANCZOS)
            s_score = ssim(crop_img, crop_img_base_rgb)
            if s_score < ssim_t:
                #print("图像ssim相似度:", s_score, " 跳过")
                continue
            #print("图像ssim相似度:", s_score, " 不跳过")
            # Mark points within the bounding box as selected
            for pt in points_raw_copy:
                if (n_x_min < int(pt[0]) < n_x_max and n_y_min < int(pt[1]) < n_y_max):
                    selected_points.append(pt)

            current_points_copy.append(point)
            current_scores_copy.append(scores_005[i])
            img_n = draw_point(points_raw, i, img_n)
    end = time.time()
    #print("adapative time:", end - start)
    add_time.append(end-start)

    draw = ImageDraw.Draw(img_n)
    draw.rectangle(EXEMPLAR_BBX, outline='red', width=2)
    return img_n, len(current_points_copy), current_points_copy,current_scores_copy


def interactive_adaptation_boxs_del(img_n, EXEMPLAR_BBX, Image_path, points_005,scores_005, current_points,current_scores,
                                    Display_width, Display_height, Image_Ori_W, Image_Ori_H, ssim_t=0.5):
    """Delete points interactively within specified bounding boxes."""
    start = time.time()
    def resize_points(points):
        """Resize points to match display dimensions."""
        return [
            (round(int(point[0]) * Display_width // Image_Ori_W), round(int(point[1]) * Display_height // Image_Ori_H))
            for point in points
        ]

    def draw_point(points, index, img):
        """Draw a single point on the image."""
        draw = ImageDraw.Draw(img)
        draw.ellipse((points[index][0] - 2, points[index][1] - 2, points[index][0] + 2, points[index][1] + 2),
                     width=2, outline='red', fill=(255, 0, 0))
        return img

    def get_area_points(x_len, y_len, points_raw,current_scores_copy, point, selected_list):
        """Get points within a specific area defined by bounding box dimensions."""
        pnum, plist,slist = 0, [], []
        for i,p in enumerate(points_raw):
            if (point[0] - x_len // 2 < int(p[0]) < point[0] + x_len // 2 and
                    point[1] - y_len // 2 < int(p[1]) < point[1] + y_len // 2 and
                    p not in selected_list):
                pnum += 1
                plist.append(p)
                slist.append(current_scores_copy[i])
        return pnum, plist, slist


    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min
    crop_area_base = (x_min, y_min, x_max, y_max)
    input_image = Image.open(Image_path).convert('RGB').resize((Display_width, Display_height), Image.LANCZOS)
    # crop_img_base = input_image.crop(crop_area_base).convert('L')
    # print(crop_area_base)
    # x_min, y_min, x_max, y_max = crop_area_base
    # if (x_min >= 0 and y_min >= 0 and
    #         x_max <= input_image.width and y_max <= input_image.height):
    #     print("裁剪区域有效")
    # else:
    #     print("裁剪区域超出图像范围")
    crop_img_base = input_image.crop(crop_area_base).convert('RGB')
    # plt.imshow(crop_img_base)
    # plt.show()  # 显示图片
    img_array_n = np.array(crop_img_base).reshape(1, -1)
    max_pixel = max(img_array_n[0])
    current_points_copy = current_points[:]
    current_scores_copy = current_scores[:]
    selected_list = []
    # start = time.time()
    #for i, point in enumerate(points_raw):
    for i,point in enumerate(current_points_copy[:]):
        pnum, plist, slist = get_area_points(x_len, y_len, current_points_copy,current_scores_copy, point, selected_list)

        if pnum > 1:
            #print("pnum:",pnum,"删除点，并更新已删除点，后面作为判断，防止重复删除")
            n_x_min, n_x_max = point[0] - x_len // 2, point[0] + x_len // 2
            n_y_min, n_y_max = point[1] - y_len // 2, point[1] + y_len // 2
            crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
            # crop_img = input_image.crop(crop_area).convert('L')
            crop_img = input_image.crop(crop_area).convert('RGB')
            if SSIM_MODE != "rgb": continue;
            if crop_img.size != crop_img_base.size:
                crop_img = crop_img.resize(crop_img_base.size, Image.LANCZOS)

            if SSIM_MODE == "rgb":
                s_score = ssim_rgb(crop_img, crop_img_base)
            elif SSIM_MODE == "avg":
                s_score = ssim_avg(crop_img, crop_img_base)
            elif SSIM_MODE == "gray":
                s_score = ssim_gray(crop_img, crop_img_base)
            else:
                raise ValueError(f"无效的SSIM模式: {SSIM_MODE}. 请使用 'rgb' 或 'avg'或 'gray'")
            #s_score = ssim(crop_img, crop_img_base)
            if s_score < ssim_t:
                #print("图像ssim相似度:", s_score, " 跳过")
                continue
            #print("图像ssim相似度:", s_score, "不跳过")
            combined = list(zip(plist, slist))
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
            sorted_plist = [item[0] for item in sorted_combined]
            sorted_slist = [item[1] for item in sorted_combined]
            # Mark and delete redundant points
            selected_list.append(sorted_plist[0])
            for i,redundant_point in enumerate(sorted_plist[1:]):
                current_points_copy.remove(redundant_point)
                current_scores_copy.remove(sorted_slist[i+1])
                draw = ImageDraw.Draw(img_n)
                draw.ellipse(
                    (redundant_point[0] - 2, redundant_point[1] - 2, redundant_point[0] + 2, redundant_point[1] + 2),
                    width=1, outline='white', fill=(255, 255, 255))
            # selected_list.append(plist[0])
            # for i, redundant_point in enumerate(plist[1:]):
            #     current_points_copy.remove(redundant_point)
            #     current_scores_copy.remove(slist[i + 1])
            #     draw = ImageDraw.Draw(img_n)
            #     draw.ellipse(
            #         (redundant_point[0] - 2, redundant_point[1] - 2, redundant_point[0] + 2, redundant_point[1] + 2),
            #         width=1, outline='black', fill=(0, 0, 0))
    end = time.time()
    #print("adapative time:", end - start)
    del_re_time.append(end-start)
    # Draw bounding box on the image
    draw = ImageDraw.Draw(img_n)
    draw.rectangle(EXEMPLAR_BBX, outline='white', width=2)
    return img_n, len(current_points_copy), current_points_copy,current_scores_copy

def interactive_adaptation_boxs_del_all(img_n, EXEMPLAR_BBX, Image_path, points_005, scores_005,current_points,current_scores,
                                    Display_width, Display_height, Image_Ori_W, Image_Ori_H, ssim_t=0.5):
    start = time.time()
    def resize_points(points):
        return [
            (round(int(point[0]) * Display_width // Image_Ori_W), round(int(point[1]) * Display_height // Image_Ori_H))
            for point in points
        ]

    def draw_point(points, index, img):
        draw = ImageDraw.Draw(img)
        draw.ellipse((points[index][0] - 2, points[index][1] - 2, points[index][0] + 2, points[index][1] + 2),
                     width=2, outline='red', fill=(255, 0, 0))
        return img

    def get_area_points(x_len, y_len, points_raw,current_scores_copy, point, selected_list):
        pnum, plist,slist = 0, [], []
        for i,p in enumerate(points_raw):
            if (point[0] - x_len // 2 < int(p[0]) < point[0] + x_len // 2 and
                    point[1] - y_len // 2 < int(p[1]) < point[1] + y_len // 2 and
                    p not in selected_list):
                pnum += 1
                plist.append(p)
                slist.append(current_scores_copy[i])
        return pnum, plist, slist

    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min
    crop_area_base = (x_min, y_min, x_max, y_max)
    input_image = Image.open(Image_path).convert('RGB').resize((Display_width, Display_height), Image.LANCZOS)
    # crop_img_base = input_image.crop(crop_area_base).convert('L')
    crop_img_base = input_image.crop(crop_area_base).convert('RGB')

    img_array_n = np.array(crop_img_base).reshape(1, -1)
    max_pixel = max(img_array_n[0])
    current_points_copy = current_points[:]
    current_scores_copy = current_scores[:]
    selected_list = []

    # start = time.time()
    for i, point in enumerate(current_points_copy[:]):
        pnum, plist, slist = get_area_points(x_len, y_len, current_points_copy, current_scores_copy, point,
                                             selected_list)

        if pnum > 1:
            n_x_min, n_x_max = point[0] - x_len // 2, point[0] + x_len // 2
            n_y_min, n_y_max = point[1] - y_len // 2, point[1] + y_len // 2
            crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
            # crop_img = input_image.crop(crop_area).convert('L')
            crop_img = input_image.crop(crop_area).convert('RGB')


            if crop_img.size != crop_img_base.size:
                crop_img = crop_img.resize(crop_img_base.size, Image.LANCZOS)

            s_score = ssim(crop_img, crop_img_base)
            if s_score < ssim_t:
                continue
            #selected_list.append(plist[0])
            #for redundant_point in plist[1:]:
            for i,redundant_point in enumerate(plist[:]):
                current_points_copy.remove(redundant_point)
                current_scores_copy.remove(slist[i])
                draw = ImageDraw.Draw(img_n)
                draw.ellipse(
                    (redundant_point[0] - 2, redundant_point[1] - 2, redundant_point[0] + 2, redundant_point[1] + 2),
                    width=1, outline='black', fill=(0, 0, 0))
    end = time.time()
    #print("adapative time:", end - start)
    del_re_time.append(end-start)
    # Draw bounding box on the image
    draw = ImageDraw.Draw(img_n)
    draw.rectangle(EXEMPLAR_BBX, outline='white', width=2)
    return img_n, len(current_points_copy), current_points_copy,current_scores_copy

def HungarianMatch(p_gt,p_prd,threshold=0.5):
    cost_martix_nap = []
    tree = KDTree(p_gt)  # 全部预测的点构建树
    for b in p_gt:
        templist = []
        ind = tree.query(b, k=4)
        d_knn = round(np.mean(ind[0][1:]))
        # print("d_knn:",d_knn)
        # print(ind[0][1:])
        # print(np.mean(ind[0][1:]))
        for a in p_prd:
            d = round(np.sqrt(np.power((np.array(a) - np.array(b)), 2).sum()))
            templist.append(round(d / d_knn, 2))
        # print(min(templist))
        cost_martix_nap.append(templist)
    # 进行匈牙利算法匹配
    row_ind, col_ind = linear_sum_assignment(cost_martix_nap)
    cost_martix = np.array(cost_martix_nap)
    return int(len(cost_martix[row_ind, col_ind][np.where(cost_martix[row_ind, col_ind] < threshold)]))


def pointF1_score(TP,p_gt,p_prd):
    FP = int(len(p_prd)) - TP
    FN = int(len(p_gt)) - TP
    #TN = anchornum - int(len(p_gt)) - FP
    if TP + FP == 0:
        Precision = 0.0
    else:
        Precision = TP / (TP + FP)  # 在预测为正样本中，实际为正样本的概率
    if TP + FN == 0:
        Recall = 0.0
    else:
        Recall = TP / (TP + FN)  # 在实际为正样本中，预测为正样本的概率
        # 防止除数为零 - F1分数
        if Precision + Recall == 0:
            F1s = 0.0
        else:
            F1s = 2 * (Precision * Recall) / (Precision + Recall)
    return F1s, Precision, Recall


def adaptation_boxes(inf, model, device, transform, root_path, ssim_t, t_view, t_candidate, t_intensity, mode):
    #root_path = "/media/xdu/Data/zrj/MYP2PNET_ROOT/crowd_datasets/CELLSsplit_64/DATA_ROOT/testval"  # 有gt 134

    Display_width = 640
    Display_height = 640
    #print(inf)
    img_name = inf['filename']
    #print(img_name)
    boxes = []
    if 'del_all' in inf['boxes']:
        boxs_del_all = inf['boxes']['del_all']
        boxes.append('del_all:'+str(boxs_del_all))
        #print('del_all:',boxs_del_all)
    if 'add' in inf['boxes']:
        boxs_add = inf['boxes']['add']
        boxes.append('add:'+str(boxs_add))
    if 'del_re' in inf['boxes']:
        boxs_del_re = inf['boxes']['del_re']
        boxes.append('del_re:'+str(boxs_del_re))

    # if 'del_all' in inf['boxes']:
    #     boxs_del_all = inf['boxes']['del_all']
    #     boxes.append('del_all:'+str(boxs_del_all))
    #     #print('del_all:',boxs_del_all)
    # if 'del_re' in inf['boxes']:
    #     boxs_del_re = inf['boxes']['del_re']
    #     boxes.append('del_re:'+str(boxs_del_re))
    # if 'add' in inf['boxes']:
    #     boxs_add = inf['boxes']['add']
    #     boxes.append('add:'+str(boxs_add))
    #print(img_name,boxes)
    # 定义正则表达式提取框类型和坐标
    box_pattern = re.compile(r"(\w+):\[(.*?)\]")
    # 提取框数据
    box_matches = box_pattern.findall(str(boxes))#640x640
    # 输出每种框的类型和坐标
    img_path = root_path+"/images/"+img_name
    print(img_path)
    img_raw = Image.open(img_path).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    # pre-proccessing
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)  # print(samples.shape)
    # samples = torch.Tensor(img).unsqueeze(0)
    #samples = samples.to(device)  # print(samples.shape)
    # run inference
    #outputs, simifeat = model(samples)  # 原文是outputs = model(samples)，但改了p2pnet的forward的return
    outputs = model(samples)  # 原文是outputs = model(samples)，但改了p2pnet的forward的return
    outputs_points = outputs['pred_points'][0]
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][
        0]  # [:, :, 1][0]为错误点的概率
    points_05 = outputs_points[outputs_scores > t_view].detach().cpu().numpy().tolist()
    points_005 = outputs_points[outputs_scores > t_candidate].detach().cpu().numpy().tolist()
    scores_05 = outputs_scores[outputs_scores > t_view].detach().cpu().numpy().tolist()
    scores_005 = outputs_scores[outputs_scores > t_candidate].detach().cpu().numpy().tolist()
    current_scores = scores_05
    current_points = points_05#原始尺寸
    def resize_points(points):
        """Resize points to match display dimensions."""
        return [
            (round(int(point[0]) * Display_width // new_width), round(int(point[1]) * Display_height // new_height))
            for point in points
        ]
    current_points = resize_points(current_points)#提前resize，在内部resize会重复

    #gt_points = points_05
    inti_points = points_05
    # 打开图片
    image = Image.open(img_path)
    img_n = image.copy()
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    # 计算缩放比例
    scale_x = image_width / 640
    scale_y = image_height / 640
    for i, point in enumerate(points_05):
        def draw_point(points, index, img):
            """Draw a single point on the image."""
            draw = ImageDraw.Draw(img)
            draw.ellipse((points[index][0] - 2, points[index][1] - 2, points[index][0] + 2, points[index][1] + 2),
                         width=12, outline="#db4bda", fill="#db4bda")
            return img
        img_n = draw_point(points_05, i, img_n)#原始尺寸
    img_n = img_n.resize((640,640))
    for box_type, coordinates in box_matches:
        # 将坐标字符串转换为实际的元组
        def convert_to_tuple_list(data):
            # 判断是否是列表且长度为4的单一列表
            if isinstance(data, list) and len(data) == 4 and all(isinstance(i, int) for i in data):
                return [tuple(data)]  # 如果是普通列表，转换为元组并包装成嵌套列表[406, 410, 435, 434] → [(406, 410, 435, 434)]
            elif isinstance(data, list) and all(isinstance(i, tuple) for i in data):
                return data  # 如果已经是嵌套的元组列表，直接返回
            else:
                return [tuple(data)]  # 其他情况，转换为元组并包装成列表
        coords = convert_to_tuple_list(list(eval(coordinates)))
        print(coords)
        # # 缩放后的坐标
        # scaled_boxes = [
        #     (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))
        #     for (x1, y1, x2, y2) in coords
        # ]
        if mode == 'PE':
            # PE模式只执行add操作
            if box_type == "add":
                print(f"  {box_type} boxes: {coords}")
                for box in coords:
                    img_n, current_count, current_points, current_scores = \
                        interactive_adaptation_boxs_add(img_n, box,
                                                        img_path, points_005, scores_005,
                                                        current_points, current_scores,
                                                        Display_width, Display_height,
                                                        image_width, image_height,
                                                        ssim_t=ssim_t, t_intensity=t_intensity)
        elif mode == 'PF':
            # PF模式只执行del_re和del_all操作
            if box_type == "del_re":
                print(f"  {box_type} boxes: {coords}")
                for box in coords:
                    img_n, current_count, current_points, current_scores = \
                        interactive_adaptation_boxs_del(img_n, box,
                                                        img_path, points_005, scores_005,
                                                        current_points, current_scores,
                                                        Display_width, Display_height,
                                                        image_width, image_height,
                                                        ssim_t=ssim_t)
            elif box_type == "del_all":
                print(f"  {box_type} boxes: {coords}")
                for box in coords:
                    img_n, current_count, current_points, current_scores = \
                        interactive_adaptation_boxs_del_all(img_n, box,
                                                            img_path, points_005, scores_005,
                                                            current_points, current_scores,
                                                            Display_width, Display_height,
                                                            image_width, image_height,
                                                            ssim_t=ssim_t)
        elif mode == 'PE_PF':
            # PE_PF模式执行所有三种操作
            if box_type == "add":
                print(f"  {box_type} boxes: {coords}")
                for box in coords:
                    img_n, current_count, current_points, current_scores = \
                        interactive_adaptation_boxs_add(img_n, box,
                                                        img_path, points_005, scores_005,
                                                        current_points, current_scores,
                                                        Display_width, Display_height,
                                                        image_width, image_height,
                                                        ssim_t=ssim_t, t_intensity=t_intensity)
            elif box_type == "del_re":
                print(f"  {box_type} boxes: {coords}")
                for box in coords:
                    img_n, current_count, current_points, current_scores = \
                        interactive_adaptation_boxs_del(img_n, box,
                                                        img_path, points_005, scores_005,
                                                        current_points, current_scores,
                                                        Display_width, Display_height,
                                                        image_width, image_height,
                                                        ssim_t=ssim_t)
            elif box_type == "del_all":
                print(f"  {box_type} boxes: {coords}")
                for box in coords:
                    img_n, current_count, current_points, current_scores = \
                        interactive_adaptation_boxs_del_all(img_n, box,
                                                            img_path, points_005, scores_005,
                                                            current_points, current_scores,
                                                            Display_width, Display_height,
                                                            image_width, image_height,
                                                            ssim_t=ssim_t)
                #draw.rectangle(box, outline="blue", width=3)
    #image.show()
    #img_n.save("box_"+img_name)

    def re_resize_points(points):
        """Resize points to match display dimensions."""
        return [
            (round(int(point[0]) *  new_width//Display_width), round(int(point[1]) *  new_height//Display_height))
            for point in points
        ]
    current_points = re_resize_points(current_points)#提前resize，在内部resize会重复


    print("inti_count",len(inti_points))
    print("current_count",len(current_points))
    # 得到gt的点坐标，不只是count
    gt_path_root = root_path + "/test_file"
    gt_path = gt_path_root + "/" + img_name.split('.')[0] + ".txt"
    with open(gt_path, 'r') as f:
        points = f.readlines()
        # print(points)
        p_gt = [[int(point.split(" ")[0]), int(point.split(" ")[1].strip())] for point in
                points]
        print("len(p_gt):", len(p_gt))

    mae_ori = abs(float(len(p_gt))-float(len(inti_points)))
    mae_cur = abs(float(len(p_gt))-float(len(current_points)))
    mse_ori = mae_ori*mae_ori
    mse_cur = mae_cur*mae_cur
    TP = HungarianMatch(p_gt, inti_points, threshold=0.5)
    F1point_score_ori, Precision_score_ori, Recall_score_ori = pointF1_score(TP, p_gt, inti_points)
    print("F1point_score_ori{},Precision_score_ori{},Recall_score_ori{}".format(F1point_score_ori,Precision_score_ori,Recall_score_ori))
    TP = HungarianMatch(p_gt, current_points, threshold=0.5)
    F1point_score_cur, Precision_score_cur, Recall_score_cur = pointF1_score(TP, p_gt, current_points)
    print("F1point_score_cur{},Precision_score_cur{},Recall_score_cur{}".format(F1point_score_cur,Precision_score_cur,Recall_score_cur))
    return mae_ori,mse_ori,F1point_score_ori,Precision_score_ori,Recall_score_ori,mae_cur,mse_cur,F1point_score_cur,Precision_score_cur,Recall_score_cur

def threeboxes_simulation(model, device, transform, args, log_path, root_path, ssim_t, t_view, t_candidate, t_intensity, mode):#用户真实交互，有三种交互，0-2次
    # model, device, transform,args = p2p_init_visual_counter()
    #ssim_t = args.ssim_t
    maes_ori, mses_ori = [], []
    m_F1point_scores_ori, m_Precision_scores_ori, m_Recall_scores_ori = [], [], []
    maes_cur, mses_cur = [], []
    m_F1point_scores_cur, m_Precision_scores_cur, m_Recall_scores_cur = [], [], []
    # 用来存储所有解析出来的框坐标及对应的文件名
    boxes_list = []
    # 定义正则表达式来匹配文件名和框的坐标（提取元组）
    # file_pattern = re.compile(r"IMG_\d+\.tif")
    file_pattern = re.compile(r"\d+\.png")
    box_pattern_del_all = re.compile(r"#sized_box_del_all:\((\d+), (\d+), (\d+), (\d+)\)")
    box_pattern_del_re = re.compile(r"#sized_box_del_re:\((\d+), (\d+), (\d+), (\d+)\)")
    box_pattern_add = re.compile(r"#sized_box_add:\((\d+), (\d+), (\d+), (\d+)\)")
    # 打开文件并逐行解析
    if log_path:
        with open(log_path, 'r', encoding='utf-8') as filetemp:
            interinf = filetemp.readlines()
            num = (len(interinf))
        with open(log_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 提取文件名
                file_match = file_pattern.search(line)
                if file_match:
                    filename = file_match.group(0)
                else:
                    continue  # 如果没有找到文件名，则跳过这一行
                boxes = {}
                # 提取框坐标数据
                del_all_matches = box_pattern_del_all.findall(line)
                del_re_matches = box_pattern_del_re.findall(line)
                add_matches = box_pattern_add.findall(line)
                # 如果有 del_all 数据，添加到字典中
                if del_all_matches:
                    boxes['del_all'] = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in del_all_matches]
                # 如果有 del_re 数据，添加到字典中
                if del_re_matches:
                    boxes['del_re'] = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in del_re_matches]
                # 如果有 add 数据，添加到字典中
                if add_matches:
                    boxes['add'] = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in add_matches]
                if boxes:
                # boxes_list.append({'filename': filename, 'boxes': boxes})
                    boxes = {'filename': filename, 'boxes': boxes}
                else:
                    boxes = {'filename': filename, 'boxes': []}
                boxes_list.append(boxes)

                mae_ori, mse_ori, F1point_score_ori, Precision_score_ori, Recall_score_ori, \
                mae_cur, mse_cur, F1point_score_cur, Precision_score_cur, Recall_score_cur = \
                    adaptation_boxes(boxes, model, device, transform, root_path, ssim_t, t_view, t_candidate, t_intensity,mode)

                m_F1point_scores_ori.append(F1point_score_ori)
                m_Precision_scores_ori.append(Precision_score_ori)
                m_Recall_scores_ori.append(Recall_score_ori)
                maes_ori.append(mae_ori)
                mses_ori.append(mse_ori)
                m_F1point_scores_cur.append(F1point_score_cur)
                m_Precision_scores_cur.append(Precision_score_cur)
                m_Recall_scores_cur.append(Recall_score_cur)
                maes_cur.append(mae_cur)
                mses_cur.append(mse_cur)
    mae_ori = np.mean(maes_ori)
    mse_ori = np.mean(mses_ori)
    m_F1point_score_ori = np.mean(m_F1point_scores_ori)
    m_Precision_score_ori = np.mean( m_Precision_scores_ori)
    m_Recall_score_ori = np.mean(m_Recall_scores_ori)

    mae_cur = np.mean(maes_cur)
    mse_cur = np.mean(mses_cur)
    m_F1point_score_cur = np.mean(m_F1point_scores_cur)
    m_Precision_score_cur = np.mean( m_Precision_scores_cur)
    m_Recall_score_cur = np.mean(m_Recall_scores_cur)

    print("mae_ori{},mse_ori{},m_F1point_scores_ori{},m_Precision_scores_ori{},m_Recall_scores_ori{}"
          .format(mae_ori,mse_ori,m_F1point_score_ori,m_Precision_score_ori,m_Recall_score_ori))
    print("mae_cur{},mse_cur{},m_F1point_scores_cur{},m_Precision_scores_cur{},m_Recall_scores_cur{}"
          .format(mae_cur,mse_cur,m_F1point_score_cur,m_Precision_score_cur,m_Recall_score_cur))
    # 记录日志
    run_log_path = "MAE_MSE_F1_P_R_scores_boxes.txt"
    with open(run_log_path, "a") as log_file:
        # 记录基础信息
        log_file.write(f'\nEval Log {time.strftime("%c")}\n')
        log_file.write(f"Config: {args}\n")
        log_file.write(f"Images: {len(interinf)}, "
                       f"SSIM_mode={SSIM_MODE}, "
                       f"ssim_t={ssim_t}, "
                       f"t_view={t_view}, "
                       f"t_candidate={t_candidate},"
                       f"t_intensity={t_intensity},"
                       f"mode={mode}\n")  # 添加模式参数

        # 单行记录原始模型性能指标
        log_file.write(
            f"ORIGINAL: F1={m_F1point_score_ori}, "
            f"Precision={m_Precision_score_ori}, "
            f"Recall={m_Recall_score_ori}, "
            f"MAE={mae_ori}, "
            f"MSE={mse_ori},"
            f"RMSE={math.sqrt(mse_ori)}\n"
        )

        # 单行记录当前模型性能指标
        log_file.write(
            f"CURRENT: F1={m_F1point_score_cur}, "
            f"Precision={m_Precision_score_cur}, "
            f"Recall={m_Recall_score_cur}, "
            f"MAE={mae_cur}, "
            f"MSE={mse_cur},"
            f"RMSE={math.sqrt(mse_cur)}\n"
        )
    # with open(run_log_path, "a") as log_file:
    #     log_file.write(f'\nEval Log {time.strftime("%c")}\n')
    #     log_file.write(f"{args}\n")
    #     log_file.write(f"m_F1point_score_ori: {m_F1point_score_ori}\n")
    #     log_file.write(f"m_Precision_score_ori: {m_Precision_score_ori}\n")
    #     log_file.write(f"m_Recall_score_ori: {m_Recall_score_ori}\n")
    #     log_file.write(f"mae_ori: {mae_ori}, mse_ori: {mse_ori}\n")
    #
    #     log_file.write(f"m_F1point_score_cur: {m_F1point_score_cur}\n")
    #     log_file.write(f"m_Precision_score_cur: {m_Precision_score_cur}\n")
    #     log_file.write(f"m_Recall_score_cur: {m_Recall_score_cur}\n")
    #     log_file.write(f"mae_cur: {mae_cur}, mse_cur: {mse_cur}\n")
    #
    #     log_file.write(f"num of img: {len(interinf)}, ssim_{SSIM_MODE},ssim_t: {ssim_t}\n")
    #     # log_file.write(
    #     #     f"num of del_re_times: {len(del_re_time)}, max_time: {np.max(del_re_time)}, mean_time: {np.mean(del_re_time)}\n")
    #     # log_file.write(
    #     #     f"num of add_times: {len(add_time)}, max_time: {np.max(add_time)}, mean_time: {np.mean(add_time)}\n")

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    # parser.add_argument('--weight_path', default='/home/hp/zrj/prjs/pth/NEFCell_best_e1500.pth',
    #                     help='path where the trained weights saved')
    parser.add_argument('--weight_path', default='/home/hp/zrj/prjs/pth/APGCC_BCD_best.pth',
                        help='path where the trained weights saved')
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--ssim_t_start', default=0.8, type=float,
                        help="start value for ssim threshold range")
    parser.add_argument('--ssim_t_end', default=0.8, type=float,
                        help="end value for ssim threshold range")
    parser.add_argument('--ssim_t_step', default=0.1, type=float,
                        help="step size for ssim threshold range")

    parser.add_argument('--t_view_start', default=0.1, type=float,
                        help="Start value for t_view range")
    parser.add_argument('--t_view_end', default=1, type=float,
                        help="End value for t_view range")
    parser.add_argument('--t_view_step', default=0.1, type=float,
                        help="Step size for t_view range")

    parser.add_argument('--t_candidate_start', default=0.01, type=float,
                        help="Start value for t_candidate range")
    parser.add_argument('--t_candidate_end', default=0.25, type=float,
                        help="End value for t_candidate range")
    parser.add_argument('--t_candidate_step', default=0.04, type=float,
                        help="Step size for t_candidate range")
    # 添加新的强度阈值参数
    parser.add_argument('--t_intensity_start', default=5, type=float,
                        help="Start value for t_candidate range")
    parser.add_argument('--t_intensity_end', default=40, type=float,
                        help="End value for t_candidate range")
    parser.add_argument('--t_intensity_step', default=5, type=float,
                        help="Step size for t_candidate range")
    # dataset parameters
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')
    return parser
def main(args):
    #与非交互式计数方法对比,初始预测模型选择训练1500个epoch的
    log_path = '/home/hp/zrj/prjs/AICC/BCD_interact_box_log.txt'
    root_path = '/home/hp/zrj/prjs/MYP2PNET_ROOT/crowd_datasets/BC_DATASET/DATA_ROOT/test'  # 3090
    # 定义要测试的模式列表
    modes = ['PE', 'PF', 'PE_PF']
    mode = 'PE_PF'
    # 初始化模型（只初始化一次）
    model, device, transform, args = apgcc_init_visual_counter()  # 调用新的函数
    # 创建SSIM阈值范围
    ssim_t_values = np.arange(
        args.ssim_t_start,
        args.ssim_t_end + args.ssim_t_step,
        args.ssim_t_step
    )
    t_view_values = np.arange(
        args.t_view_start,
        args.t_view_end + args.t_view_step,
        args.t_view_step
    )

    t_candidate_values = np.arange(
        args.t_candidate_start,
        args.t_candidate_end + args.t_candidate_step,
        args.t_candidate_step
    )
    # t_candidate_values = [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.3,0.35,0.4,0.45,0.5]
    t_candidate_values = [0.05]

    t_intensity_values = np.arange(
        args.t_intensity_start,
        args.t_intensity_end + args.t_intensity_step,
        args.t_intensity_step
    )
    # 初始化模型（只初始化一次）
    # model, device, transform, args = p2p_init_visual_counter()
    #model, device, transform, args = apgcc_init_visual_counter()

    # for ssim_t in ssim_t_values:
    #     print(f"\n{'=' * 50}")
    #     print(f"Testing with SSIM threshold: {ssim_t:.2f}")
    #     print(f"{'=' * 50}")
    #     # 修改SSIM阈值并运行测试
    #     threeboxes_simulation(model, device, transform, args, log_path, root_path, ssim_t)
    # ssim_t=0.8
    # t_intensity=20
    # t_candidate = 0.05
    # for t_view in t_view_values:
    #     print(f"\n{'=' * 50}")
    #     print(f"Testing with SSIM threshold: {t_view:.2f}")
    #     print(f"{'=' * 50}")
    #     # 修改SSIM阈值并运行测试
    #     threeboxes_simulation(model, device, transform, args, log_path, root_path, ssim_t, t_view, t_candidate, t_intensity)
    ssim_t=0.8
    t_view=0.5
    t_intensity=20
    for t_candidate in t_candidate_values:
        print(f"\n{'=' * 50}")
        print(f"Testing with t_candidate threshold: {t_candidate}")
        print(f"{'=' * 50}")
        # 修改SSIM阈值并运行测试
        threeboxes_simulation(model, device, transform, args, log_path, root_path,
                          ssim_t, t_view, t_candidate, t_intensity, mode)
    # ssim_t=0.8
    # t_view=0.5
    # t_candidate = 0.05
    # for t_intensity in t_intensity_values:
    #     print(f"\n{'=' * 50}")
    #     print(f"Testing with t_intensity threshold: {t_intensity:.2f}")
    #     print(f"{'=' * 50}")
    #     # 修改SSIM阈值并运行测试
    #     threeboxes_simulation(model, device, transform, args, log_path, root_path,
    #                           ssim_t, t_view, t_candidate, t_intensity, mode)
    # ssim_t=0.8
    # t_view=0.5
    # t_candidate = 0.05
    # t_intensity = 20
    # for mode in modes:
    #     print(f"\n{'=' * 50}")
    #     print(f"Testing with mode: {mode}")
    #     print(f"{'=' * 50}")
    #     # 修改SSIM阈值并运行测试
    #     threeboxes_simulation(model, device, transform, args, log_path, root_path,
    #                           ssim_t, t_view, t_candidate, t_intensity, mode)
if __name__ == '__main__':
    # parser = argparse.ArgumentParser('P2PNet_simulation', parents=[get_args_parser()])
    # args = parser.parse_args()
    # main(args)
    # 这一部分不再需要，因为 apgcc_init_visual_counter 内部会自己处理
    # parser = argparse.ArgumentParser('APGCC_simulation', parents=[get_apgcc_args_parser()])
    # args = parser.parse_args()

    # 直接调用 main 函数，main 函数内部会调用 apgcc_init_visual_counter 来完成所有初始化
    main(None)  # 可以传 None，因为 main 函数内部不再直接依赖这个输入参数