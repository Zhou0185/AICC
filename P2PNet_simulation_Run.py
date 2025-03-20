import argparse
import os
import pdb
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
from models import build_model
from sklearn.cluster import KMeans

# pdb.set_trace()
del_re_time = []
add_time = []
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
def p2p_init_visual_counter():
    parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    print(args)
    device = torch.device('cuda:{}'.format(args.gpu_id))
    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        # convert to eval mode
        model.eval()
    # create the pre-processing transform
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model,device,transform,args

def ssim(y_true, y_pred):
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
    ssim_channels.remove(max(ssim_channels))
    return np.mean(ssim_channels)

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

def dataplot_v3_drawbox(inf,model,device,transform):
    #log_path = "/media/xd/zrj/MyData/test_interact_v3/interact_log_Cellsplitv3lastb4e2000.txt"
    root_path = "/media/xd/zrj/Prjs/MYP2PNET_ROOT/crowd_datasets/CELLSsplit_v3/DATA_ROOT/test"
    Display_width = 640
    Display_height = 640

    print(inf)#得到box坐标，图片名称
    ind_box = inf.find("sized_interact_box_area")
    box_area = inf[ind_box:].split("(")[1].split(")")[0].split(",")

    ind_gtc = inf.find("gt_count:")
    gt_count = int(inf[ind_gtc:].split(":")[1].split(" ")[0])

    img_name = inf.split("images/")[1].split(" ")[0]
    #print(img_name)
    Image_path = root_path+"/images/"+img_name
    img_raw = Image.open(Image_path).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)  # print(samples.shape)
    # run inference
    threshold = 0.5
    outputs, simifeat = model(samples)  # 原文是outputs = model(samples)，但改了p2pnet的forward的return
    outputs_points = outputs['pred_points'][0]
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][
        0]  # [:, :, 1][0]为错误点的概率
    points_05 = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    points_005 = outputs_points[outputs_scores > 0.05].detach().cpu().numpy().tolist()
    img_ini = DrawfPoints(points_05,Image_path)
    predict_cnt = len(points_05)
    print(predict_cnt)

    #坐标还原原图尺寸
    EXEMPLAR_BBX_ori = (int(box_area[0].strip()),int(box_area[1].strip()),
                  int(box_area[2].strip()),int(box_area[3].strip()))
    EXEMPLAR_BBX=(int(box_area[0].strip())*width/Display_width,int(box_area[1].strip())*width/Display_width,
                  int(box_area[2].strip())*height/Display_height,int(box_area[3].strip())*height/Display_height)
    print(EXEMPLAR_BBX_ori)
    img_n = Image.fromarray(np.uint8(img_ini))
    draw = ImageDraw.Draw(img_n)
    draw.rectangle(EXEMPLAR_BBX, outline='red', width=2)
    # plt.imshow(img_n)
    # plt.show()  # 显示图片

    #得到gt的点坐标，不只是count
    gt_path_root = root_path + "/test_file"
    gt_path = gt_path_root + "/" + img_name.split('.')[0] + ".txt"
    with open(gt_path, 'r') as f:
        points = f.readlines()
        # print(points)
        p_gt = [[int(point.split(" ")[0]), int(point.split(" ")[1].strip())] for point in
                points]
        print("len(p_gt):", len(p_gt), p_gt)
    return img_n,EXEMPLAR_BBX_ori,Image_path,points_05,points_005,gt_count,\
           Display_width,Display_height,new_width,new_height,p_gt

def interactive_adaptation_box_add(img_n,EXEMPLAR_BBX,Image_path,points_05,points_005,gt_count,Display_width,Display_height,Image_Ori_W,Image_Ori_H):
    def SizedPoints(points):
        sizedpoints = []
        data = points
        llen = len(data)
        for i in range(llen):
            x = round(int(data[i][0]) * Display_width // Image_Ori_W)
            y = round(int(data[i][1]) * Display_height // Image_Ori_H)
            sizedpoints.append((x, y))
            # print(int(data[i][0]),int(data[i][1]),"__",x,y)
        return sizedpoints
    def DrawPointi(pointsraw,sindex,img_n):
        draw = ImageDraw.Draw(img_n)
        draw.ellipse((pointsraw[sindex][0] - 2, pointsraw[sindex][1] - 2, pointsraw[sindex][0] + 2,
                      pointsraw[sindex][1] + 2), width=2,
                     outline='red', fill=(255, 0, 0))
        return img_n
    add_points = []
    x_min = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])#self.EXEMPLAR_START_X, self.EXEMPLAR_END_X
    x_max = max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])#self.EXEMPLAR_START_X, self.EXEMPLAR_END_X
    y_min = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])#self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y
    y_max = max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])#self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y
    x_len = x_max - x_min
    y_len = y_max - y_min
    #得到框内的图像信息
    crop_area_base = (x_min, y_min, x_max, y_max)
    #print("crop_area_base",crop_area_base)
    Input_Image = Image.open(Image_path).convert('RGB').resize((Display_width,Display_height), Image.LANCZOS)
    crop_img_base = Input_Image.copy().crop(crop_area_base)
    crop_img_base = crop_img_base.convert('L')#转换成灰度图
    # plt.imshow(crop_img_base)
    # plt.show()  # 显示图片
    img_array_n = np.array(crop_img_base).reshape(1, -1)
    max_pixel = max(img_array_n[0])#得到框选图像的最大像素值
    #print("max_pixel:",max_pixel)
    # mean_pixel = np.mean(img_array_n[0])#得到框选图像的平均像素值
    # print("mean_pixel:",mean_pixel)
    #为所有点画框(置信度大于0.05)
    selectedp_list = []
    sizedpointsraw_05 = SizedPoints(points_05)
    sizedpointsraw_005 = SizedPoints(points_005)
    pointsraw = torch.tensor(sizedpointsraw_005).view(-1, 2).tolist()#画框内的全部锚点
    pointsraw1 = torch.tensor(sizedpointsraw_005).view(-1, 2).tolist()#画框内的全部锚点，用于计计算框最中间的点
    sizedpoints = torch.tensor(sizedpointsraw_05).view(-1, 2).tolist()

    img_raw = Image.open(Image_path).convert('L').resize((Display_width, Display_height), Image.LANCZOS)
    img_array = np.array(img_raw)

    # 为所有点遍历(置信度大于0.05)
    for i in range(len(pointsraw)):
        pnum = 0
        if pointsraw[i] in selectedp_list:
            #print("point selected,continue")
            continue
        if(pointsraw[i][0]>=640):
            pointsraw[i][0]=639
        if(pointsraw[i][1]>=640):
            pointsraw[i][1]=639
        #print(pointsraw[i][0],pointsraw[i][1])
        currentpoint_pixel = img_array[pointsraw[i][1]][pointsraw[i][0]]#---注意翻一下横纵坐标
        if(currentpoint_pixel<max_pixel//2):
            #print("point's pixel likes background,continue ","该点的像素值:",currentpoint_pixel," 目标参考像素值max_pixel//2：",max_pixel//2)
            continue
        for point1 in sizedpoints:#计算以点i为中心，画框范围内的的目前点个数（大于1说明该物体已被计数，不用重复计数）
            if int(point1[0]) > pointsraw[i][0] - x_len // 2 and int(point1[0]) < pointsraw[i][0] + x_len // 2 \
                    and int(point1[1]) > pointsraw[i][1] - y_len // 2 and int(point1[1]) < pointsraw[i][1] + y_len // 2:
                #print(point1)
                pnum = pnum + 1
        if pnum == 0:
            #print("该框中无预测点，选择目前点为预测点（以目前点画的框，已在框的中间）")
            n_x_min = pointsraw[i][0] - x_len // 2;
            n_x_max = pointsraw[i][0] + x_len // 2
            n_y_min = pointsraw[i][1] - y_len // 2;
            n_y_max = pointsraw[i][1] + y_len // 2
            crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
            crop_img = Input_Image.copy().crop(crop_area)
            crop_img = crop_img.convert('L')  # 转换成灰度图
            if crop_img_base.size != crop_img.size:
                crop_img = crop_img.resize((crop_img_base.size[0], crop_img_base.size[1]), Image.LANCZOS)
            s_score = ssim(crop_img, crop_img_base)
            # plt.imshow(crop_img)
            # plt.show()  # 显示图片
            if s_score < 0.5:
                #print("图像ssim相似度:", s_score," 跳过")
                continue
            #print("图像ssim相似度:", s_score, " 不跳过")

            # 选择框内最中间的点
            for j in range(len(pointsraw1)):#选择框内最中间的点,并把 框内所有 点加入队列，防止重复
                if int(pointsraw1[j][0]) > n_x_min and int(pointsraw1[j][0]) < n_x_max \
                        and int(pointsraw1[j][1]) > n_y_min and int(pointsraw1[j][1]) < n_y_max:
                        #and pointsraw[i] not in selectedp_list:
                    #print("加入selected",pointsraw1[j])
                    selectedp_list.append(pointsraw1[j])

            sizedpoints.append(pointsraw[i])
            add_points.append(pointsraw[i])
            #print(pointsraw1[sindex],'==：',pointsraw[i])
            #print('self.sizedpoints_+：',len(sizedpoints))
            img_n = img_n.resize((Display_width,Display_height), Image.LANCZOS)
            img_n = DrawPointi(pointsraw,i,img_n)
        else:
            #print("point not only in box,continue")
            continue

    print("initial_count:",len(points_05)," gt_count:",gt_count)
    print("current_count:",len(sizedpoints))
    # plt.imshow(img_n)
    # plt.show()  # 显示图片
    return int(gt_count),int(len(points_05)),int(len(sizedpoints)),sizedpoints


def interactive_adaptation_boxs_add(img_n, EXEMPLAR_BBX, Image_path, points_005,scores_005, current_points,current_scores,
                                    Display_width, Display_height, Image_Ori_W, Image_Ori_H, ssim_t=0.5):
    """Add points interactively within specified bounding boxes."""
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
                     width=12, outline='red', fill=(255, 0, 0))
        return img

    # Calculate bounding box dimensions
    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min
    # img_open = Image.open(self.Image_path)
    # self.Image_name = self.Image_path.split("/")[-1].split(".")[0]
    # gt_path = self.Image_path.replace("images", "test_file").replace(".tif", ".txt")
    # print(gt_path)
    # if os.path.exists(gt_path):
    #     self.Gt_path = gt_path
    # self.max_hw = 1504
    # W, H = img_open.size
    # img_open = img_open.resize((self.Display_width, self.Display_height))
    # img_show = ImageTk.PhotoImage(img_open)
    # self.Input_Image_Label.configure(image=img_show)
    # self.Input_Image_Label.image = img_show
    # self.Input_Image = img_open.copy()

    # Crop and process image within bounding box
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

    # 裁剪图像
    crop_img_base_rgb = input_image.crop(crop_area_base)

    # 检查裁剪图像的像素值范围
    # crop_img_array = np.array(crop_img_base_rgb)
    # print("裁剪图像像素值范围:", crop_img_array.min(), crop_img_array.max())
    #
    # # 显示裁剪图像
    # plt.imshow(crop_img_base_rgb, vmin=0, vmax=255)  # 假设像素值范围是 [0, 255]
    # plt.imsave("crop_img_base_rgb.png", crop_img_base_rgb)
    # plt.show()
    #rgb
    # 将图像转换为 numpy 数组并展平
    crop_img_base_rgb_array = np.array(crop_img_base_rgb)
    # 将图像展平成 (num_pixels, 3) 的二维数组，每行代表一个像素的 RGB 值
    pixels_rgb = crop_img_base_rgb_array.reshape(-1, 3)

    # 检查数据中的独特点
    #unique_points = np.unique(pixels_rgb, axis=0)
    #print("Unique points:", unique_points)
    # 根据独特点数量设置 n_clusters
    #nn_clusters = len(unique_points)
    #print("nn_clusters",nn_clusters)
    # 使用 K-means 进行聚类，将图像分为两类（细胞和背景）
    #kmeans_rgb = KMeans(n_clusters=n_clusters, random_state=0, n_init=10).fit(pixels_rgb)
    kmeans_rgb = KMeans(n_clusters=2, random_state=0).fit(pixels_rgb)
    # kmeans_rgb = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(pixels_rgb)
    # 获取聚类标签（0 或 1），每个标签对应一个类
    labels_rgb = kmeans_rgb.labels_
    # 将标签重新形状为原图大小
    labels_rgb_image = labels_rgb.reshape(crop_img_base_rgb_array.shape[:2])
    # 定义中心区域的大小
    center_area_size = 5  # 5x5 的中心区域
    half_size = center_area_size // 2
    # 获取图像中心坐标
    center_x, center_y = crop_img_base_rgb_array.shape[0] // 2, crop_img_base_rgb_array.shape[1] // 2
    # 获取中心区域内的标签，并统计出现频率
    center_area_labels = labels_rgb_image[
                         center_x - half_size:center_x + half_size + 1,
                         center_y - half_size:center_y + half_size + 1
                         ]
    # 找到中心区域内最频繁出现的标签作为细胞区域标签
    cell_label = np.bincount(center_area_labels.ravel()).argmax()
    # 创建细胞区域的掩码
    mask = (labels_rgb_image == cell_label).astype(np.uint8)
    # 提取细胞部分和背景部分的像素
    cell_pixels = crop_img_base_rgb_array[mask == 1]
    #background_pixels = crop_img_base_rgb_array[mask == 0]
    # 计算细胞部分和背景部分的平均像素值
    average_cell_pixel = np.mean(cell_pixels.mean(axis=0))
    #average_background_pixel = background_pixels.mean(axis=0)
    # 计算背景和细胞的平均像素差值
    #average_pixel_diff = np.mean(np.abs(average_cell_pixel - average_background_pixel))

    # Resize points
    resized_points = resize_points(points_005)
    points_raw = torch.tensor(resized_points).view(-1, 2).tolist()
    points_raw_copy = points_raw.copy()
    # current_points = resized_points(current_points)#会重复reszie
    current_points_copy = current_points[:]
    current_scores_copy = current_scores[:]

    # img_raw = Image.open(Image_path).convert('L').resize((Display_width, Display_height), Image.LANCZOS)
    img_raw = Image.open(Image_path).convert('RGB').resize((Display_width, Display_height))
    img_array = np.array(img_raw)
    selected_points = []

    # Traverse all points (confidence > 0.05)
    # start = time.time()
    for i, point in enumerate(points_raw):
        if point in selected_points:
            continue

        # Ensure points stay within bounds
        point[0] = min(point[0], Display_width - 1)
        point[1] = min(point[1], Display_height - 1)
        current_pixel = np.mean(img_array[points_raw[i][1]][points_raw[i][0]])  # ---注意翻一下横纵坐标
        #current_pixel = img_array[point[1]][point[0]]
        # if current_pixel < max_pixel // 2:
        #     continue
        #if (abs(current_pixel - average_cell_pixel) > 8):
        if (abs(current_pixel - average_cell_pixel) > 20):
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
    # Draw bounding box on the image
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

    # Calculate bounding box dimensions
    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min

    # Crop and process image within bounding box
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

    # Resize points
    # resized_points = resize_points(points_005)
    # points_raw = torch.tensor(resized_points).view(-1, 2).tolist()
    current_points_copy = current_points[:]
    current_scores_copy = current_scores[:]
    selected_list = []

    # Traverse each point in current points
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


            if crop_img.size != crop_img_base.size:
                crop_img = crop_img.resize(crop_img_base.size, Image.LANCZOS)

            s_score = ssim(crop_img, crop_img_base)
            if s_score < ssim_t:
                #print("图像ssim相似度:", s_score, " 跳过")
                continue
            #print("图像ssim相似度:", s_score, "不跳过")
            # 将plist和slist结合成一个列表，其中每个元素是一个元组，包含plist中的点和对应的分数
            combined = list(zip(plist, slist))
            # 根据分数（slist的值）降序排序
            sorted_combined = sorted(combined, key=lambda x: x[1], reverse=True)
            # 提取排序后的plist（点）
            sorted_plist = [item[0] for item in sorted_combined]
            # 提取排序后的 slist（分数）
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

    # Calculate bounding box dimensions
    x_min, x_max = min(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2]), max(EXEMPLAR_BBX[0], EXEMPLAR_BBX[2])
    y_min, y_max = min(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3]), max(EXEMPLAR_BBX[1], EXEMPLAR_BBX[3])
    x_len, y_len = x_max - x_min, y_max - y_min

    # Crop and process image within bounding box
    crop_area_base = (x_min, y_min, x_max, y_max)
    input_image = Image.open(Image_path).convert('RGB').resize((Display_width, Display_height), Image.LANCZOS)
    # crop_img_base = input_image.crop(crop_area_base).convert('L')
    crop_img_base = input_image.crop(crop_area_base).convert('RGB')

    img_array_n = np.array(crop_img_base).reshape(1, -1)
    max_pixel = max(img_array_n[0])

    # Resize points
    # resized_points = resize_points(points_005)
    # points_raw = torch.tensor(resized_points).view(-1, 2).tolist()
    current_points_copy = current_points[:]
    current_scores_copy = current_scores[:]
    selected_list = []

    # Traverse each point in current points
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

            # Mark and delete redundant points
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

def HungarianMatch(p_gt,p_prd,threshold=0.25):
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
    Precision = TP/(TP+FP)#在预测为正样本中，实际为正样本的概率
    Recell = TP/(TP+FN)#在实际为正样本中，预测为正样本的概率
    F1s = 2*(Precision*Recell)/(Precision+Recell)
    return F1s, Precision, Recell

def dataplot_v3_drawbox_3_3(inf,model,device,transform,root_path):
    root_path = root_path
    # root_path = "/mnt/disk3/zrj/MyDatas/CELLSsplit_v4/DATA_ROOT/test"#有gt 43090
    Display_width = 640
    Display_height = 640
    # if log_path:
    #     with open(log_path, "r") as f:
    #         interinf = f.readlines()
    #         for inf in interinf:
    print(inf)#得到box坐标，图片名称
    ind_box_add = inf.find("#sized_box_add")
    boxs_area_add = inf[ind_box_add:].split("[(")[1].split(")]")[0].split("), (")
    ind_box_del = inf.find("#sized_box_del")
    boxs_area_del = inf[ind_box_del:].split("[(")[1].split(")]")[0].split("), (")
    img_name = inf.split("images/")[1].split("#")[0]
    #print(img_name)
    Image_path = root_path+"/images/"+img_name
    img_raw = Image.open(Image_path).convert('RGB')
    W, H = img_raw.size
    img_n = Image.fromarray(np.uint8(img_raw))

    #img_raw = Image.open(Image_path).convert('RGB')
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)  # print(samples.shape)
    # run inference
    threshold = 0.5
    outputs, simifeat = model(samples)  # 原文是outputs = model(samples)，但改了p2pnet的forward的return
    outputs_points = outputs['pred_points'][0]
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]  # [:, :, 1][0]为错误点的概率
    points_05 = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    points_005 = outputs_points[outputs_scores > 0.05].detach().cpu().numpy().tolist()
    scores_05 = outputs_scores[outputs_scores > 0.5].detach().cpu().numpy().tolist()
    scores_005 = outputs_scores[outputs_scores > 0.05].detach().cpu().numpy().tolist()

    img_ini = DrawfPoints(points_05, Image_path)
    img_ini = Image.fromarray(np.uint8(img_ini))
    predict_cnt = len(points_05)
    print(predict_cnt)
    #坐标还原原图尺寸
    print(boxs_area_add)
    print(boxs_area_del)
    EXEMPLAR_BBX_ori_add_list = []
    EXEMPLAR_BBX_ori_del_list = []
    for i in range(len(boxs_area_add)):
        box_area_add = boxs_area_add[i].split(", ")
        EXEMPLAR_BBX_ori_add = (
        int(box_area_add[0].strip()) , int(box_area_add[1].strip()),
        int(box_area_add[2].strip()) , int(box_area_add[3].strip()))
        EXEMPLAR_BBX_add=(int(box_area_add[0].strip())*W/Display_width,int(box_area_add[1].strip())*W/Display_width,
                      int(box_area_add[2].strip())*H/Display_height,int(box_area_add[3].strip())*H/Display_height)
        #print(EXEMPLAR_BBX_add)
        EXEMPLAR_BBX_ori_add_list.append(EXEMPLAR_BBX_ori_add)

        box_area_del = boxs_area_del[i].split(", ")
        EXEMPLAR_BBX_ori_del = (
            int(box_area_del[0].strip()), int(box_area_del[1].strip()),
            int(box_area_del[2].strip()), int(box_area_del[3].strip()))
        EXEMPLAR_BBX_del = (
        int(box_area_del[0].strip()) * W / Display_width, int(box_area_del[1].strip()) * W / Display_width,
        int(box_area_del[2].strip()) * H / Display_height,int(box_area_del[3].strip()) * H / Display_height)
        #print(EXEMPLAR_BBX_del)
        EXEMPLAR_BBX_ori_del_list.append(EXEMPLAR_BBX_ori_del)

        # draw = ImageDraw.Draw(img_n)
        # draw.rectangle(EXEMPLAR_BBX_add, outline='red', width=2)
        # draw.rectangle(EXEMPLAR_BBX_del, outline='white', width=2)
    # plt.imshow(img_n)
    # plt.show()  # 显示图片
    #break

    #得到gt的点坐标，不只是count
    gt_path_root = root_path + "/test_file"
    gt_path = gt_path_root + "/" + img_name.split('.')[0] + ".txt"
    with open(gt_path, 'r') as f:
        points = f.readlines()
        # print(points)
        p_gt = [[int(point.split(" ")[0]), int(point.split(" ")[1].strip())] for point in
                points]
        #print("len(p_gt):", len(p_gt), p_gt)
    return EXEMPLAR_BBX_ori_add_list,EXEMPLAR_BBX_ori_del_list,points_05,points_005,scores_05,scores_005,p_gt,\
           img_ini,Image_path,Display_width,Display_height,new_width,new_height

def sixbox_F1(log_path,root_path):#模拟交互，3次加点，3次删除重复点
    #log_path = "/mnt/disk3/zrj/PICACount/interact_box_log_Cellsplitv4lastb4e6交互_box33.txt"#43090
    log_path = log_path
    #root_path = "/media/xd/zrj/Prjs/MYP2PNET_ROOT/crowd_datasets/CELLSsplit_v3/DATA_ROOT/test"
    model, device, transform, args = p2p_init_visual_counter()
    ssim_t = args.ssim_t
    def ReSizedPoints(points):
        resizedpoints = []
        data = points
        llen = len(data)
        for i in range(llen):
            x = round(int(data[i][0]) * Image_Ori_W // Display_width)
            y = round(int(data[i][1]) * Image_Ori_H // Display_height)
            resizedpoints.append((x, y))
            # print(int(data[i][0]),int(data[i][1]),"__",x,y)
        return resizedpoints

    maes_ori, mses_ori = [], []
    m_F1point_scores_ori, m_Precision_scores_ori, m_Recall_scores_ori = [], [], []

    # 使用列表创建三个子列表
    m_F1point_scores_cur = [[] for _ in range(3)]
    m_Precision_scores_cur = [[] for _ in range(3)]
    m_Recall_scores_cur = [[] for _ in range(3)]
    maes_cur = [[] for _ in range(3)]
    mses_cur = [[] for _ in range(3)]
    if log_path:
        with open(log_path, "r") as f:
            interinf = f.readlines()
            for inf in interinf:
                EXEMPLAR_BBX_ori_add_list, EXEMPLAR_BBX_ori_del_list, points_05,points_005,scores_05,scores_005,p_gt,\
                img_n,Image_path,Display_width,Display_height,Image_Ori_W, Image_Ori_H \
                    =  dataplot_v3_drawbox_3_3(inf, model, device, transform,root_path)

                print(Image_path,EXEMPLAR_BBX_ori_add_list, EXEMPLAR_BBX_ori_del_list)
                print("points_05,points_005,p_gt ",len(points_05),len(points_005),len(p_gt))
                gt_count = len(p_gt)
                def SizedPoints(points):
                    sizedpoints = []
                    data = points
                    llen = len(data)
                    for i in range(llen):
                        x = round(int(data[i][0]) * Display_width // Image_Ori_W)
                        y = round(int(data[i][1]) * Display_height // Image_Ori_H)
                        sizedpoints.append((x, y))
                        # print(int(data[i][0]),int(data[i][1]),"__",x,y)
                    return sizedpoints
                sizedpointsraw_05 = SizedPoints(points_05)
                sizedpoints = torch.tensor(sizedpointsraw_05).view(-1, 2).tolist()
                current_points = sizedpoints
                current_scores = scores_05
                initial_count = len(sizedpoints)
                threshold = 0.5
                # threshold = 10#无knn
                TP = HungarianMatch(p_gt, points_05,threshold)

                F1point_score_ori, Precision_score_ori, Recall_score_ori = pointF1_score(TP, p_gt, points_05)
                maes_ori.append(abs(initial_count - gt_count))
                mses_ori.append(abs(initial_count - gt_count) ** 2)
                for i in range(len(EXEMPLAR_BBX_ori_add_list)):
                    img_n, current_count, current_points,current_scores \
                        = interactive_adaptation_boxs_add(img_n, EXEMPLAR_BBX_ori_add_list[i],
                                                         Image_path, points_005,scores_005,current_points,current_scores,
                                                         Display_width, Display_height,
                                                         Image_Ori_W, Image_Ori_H,
                                                         ssim_t)
                    print("initial_count, add_current_count:", initial_count, current_count)
                    img_n, current_count, current_points,current_scores \
                        = interactive_adaptation_boxs_del(img_n, EXEMPLAR_BBX_ori_del_list[i],
                                                          Image_path, points_005,scores_005, current_points,current_scores,
                                                          Display_width, Display_height,
                                                          Image_Ori_W, Image_Ori_H,
                                                          ssim_t)
                    print("initial_count, del_current_count:", initial_count, current_count)

                    maes_cur[i].append(abs(current_count - gt_count))
                    mses_cur[i].append(abs(current_count - gt_count) ** 2)

                    resizedpoints = ReSizedPoints(current_points)
                    TP = HungarianMatch(p_gt, resizedpoints,threshold)
                    F1point_score_cur, Precision_score_cur, Recall_score_cur = pointF1_score(TP, p_gt, resizedpoints,
                                                                                            )
                    print("F1point_score_ori{},F1point_score_cur{}:{}".format(F1point_score_ori,i,F1point_score_cur))
                    m_F1point_scores_cur[i].append(F1point_score_cur)
                    m_Precision_scores_cur[i].append(Precision_score_cur)
                    m_Recall_scores_cur[i].append(Recall_score_cur)
                    # plt.imshow(img_n)
                    # plt.show()  # 显示图片
                m_F1point_scores_ori.append(F1point_score_ori)
                m_Precision_scores_ori.append(Precision_score_ori)
                m_Recall_scores_ori.append(Recall_score_ori)
                #m_F1point_scores_cur.append(F1point_score_cur)
                #break

            # 计算原始分数的均值并打印
            m_F1point_score_ori = np.mean(m_F1point_scores_ori)
            m_Precision_score_ori = np.mean(m_Precision_scores_ori)
            m_Recall_score_ori = np.mean(m_Recall_scores_ori)
            mae_ori = np.mean(maes_ori)
            mse_ori = np.mean(mses_ori)
            rmse_ori = np.sqrt(mse_ori)
            print(f"m_F1point_score_ori: {m_F1point_score_ori}")
            print(f"m_Precision_score_ori: {m_Precision_score_ori}")
            print(f"m_Recall_score_ori: {m_Recall_score_ori}")
            print(f"mae_ori: {mae_ori}, mse_ori: {mse_ori}, rmse_ori: {rmse_ori}")
            # 打印当前分数的均值
            for i, (f1, precision, recall) in enumerate(
                    zip(m_F1point_scores_cur, m_Precision_scores_cur, m_Recall_scores_cur)):
                print(f"m_F1point_score_cur{i}: {np.mean(f1)}")
                print(f"m_Precision_score_cur{i}: {np.mean(precision)}")
                print(f"m_Recall_score_cur{i}: {np.mean(recall)}")
            # 打印当前 MAE 和 MSE 的均值
            for i, (mae, mse) in enumerate(zip(maes_cur, mses_cur)):
                print(f"maes_cur{i}: {np.mean(mae)}")
                print(f"mses_cur{i}: {np.mean(mse)}")
                print(f"rmses_cur{i}: {np.sqrt(np.mean(mse))}")
            # 记录日志
            run_log_path = "MAE_MSE_F1_P_R_scores_box33.txt"
            with open(run_log_path, "a") as log_file:
                log_file.write(f'\nEval Log {time.strftime("%c")}\n')
                log_file.write(f"{args}\n")
                log_file.write(f"m_F1point_score_ori: {m_F1point_score_ori}\n")
                log_file.write(f"m_Precision_score_ori: {m_Precision_score_ori}\n")
                log_file.write(f"m_Recall_score_ori: {m_Recall_score_ori}\n")
                log_file.write(f"mae_ori: {mae_ori}, mse_ori: {mse_ori}, rmse_ori: {rmse_ori}\n")
                # 记录当前分数的均值
                for i, (f1, precision, recall) in enumerate(
                        zip(m_F1point_scores_cur, m_Precision_scores_cur, m_Recall_scores_cur)):
                    log_file.write(f"m_F1point_score_cur{i}: {np.mean(f1)}\n")
                    log_file.write(f"m_Precision_score_cur{i}: {np.mean(precision)}\n")
                    log_file.write(f"m_Recall_score_cur{i}: {np.mean(recall)}\n")
                # 记录当前 MAE 和 MSE 的均值
                for i, (mae, mse) in enumerate(zip(maes_cur, mses_cur)):
                    log_file.write(f"maes_cur{i}: {np.mean(mae)}\n")
                    log_file.write(f"mses_cur{i}: {np.mean(mse)}\n")
                    log_file.write(f"rmses_cur{i}: {np.sqrt(np.mean(mse))}\n")

                log_file.write(f"num of img: {len(interinf)}\n")
                log_file.write(f"num of img: {len(interinf)}, confidence_t: {threshold}, ssim_t: {ssim_t}\n")

                log_file.write(f"num of del_re_times: {len(del_re_time)}, max_time: {np.max(del_re_time)}, mean_time: {np.mean(del_re_time)}\n")
                log_file.write(f"num of add_times: {len(add_time)}, max_time: {np.max(add_time)}, mean_time: {np.mean(add_time)}\n")#

def adaptation_boxes(inf,model,device,transform,root_path,ssim_t):
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
    img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
    # pre-proccessing
    img = transform(img_raw)
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)  # print(samples.shape)
    # run inference
    outputs, simifeat = model(samples)  # 原文是outputs = model(samples)，但改了p2pnet的forward的return
    outputs_points = outputs['pred_points'][0]
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][
        0]  # [:, :, 1][0]为错误点的概率
    points_05 = outputs_points[outputs_scores > 0.5].detach().cpu().numpy().tolist()
    points_005 = outputs_points[outputs_scores > 0.05].detach().cpu().numpy().tolist()
    scores_05 = outputs_scores[outputs_scores > 0.5].detach().cpu().numpy().tolist()
    scores_005 = outputs_scores[outputs_scores > 0.05].detach().cpu().numpy().tolist()
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
        if box_type == "add":
            print(f"  {box_type} boxes: {coords}")
            for box in coords:
                img_n, current_count, current_points,current_scores = \
                    interactive_adaptation_boxs_add(img_n, box,
                                                    img_path, points_005, scores_005,
                                                    current_points,current_scores,
                                                    Display_width, Display_height,
                                                    image_width, image_height,
                                                    ssim_t=ssim_t)
                #draw.rectangle(box, outline="white", width=3)
                # img_n, current_count, current_points, current_scores \
                #     = interactive_adaptation_boxs_add(img_n, EXEMPLAR_BBX_ori_add_list[i],
                #                                       Image_path, points_005, scores_005, current_points,
                #                                       current_scores,
                #                                       Display_width, Display_height,
                #                                       Image_Ori_W, Image_Ori_H,
                #                                       ssim_t)
        if box_type == "del_re":
            print(f"  {box_type} boxes: {coords}")
            for box in coords:
                img_n, current_count, current_points,current_scores = \
                    interactive_adaptation_boxs_del(img_n, box,
                                                    img_path, points_005, scores_005,
                                                    current_points,current_scores,
                                                    Display_width, Display_height,
                                                    image_width, image_height,
                                                    ssim_t=ssim_t)
                #draw.rectangle(box, outline="red", width=3)
                # img_n, current_count, current_points, current_scores \
                #     = interactive_adaptation_boxs_del(img_n, EXEMPLAR_BBX_ori_del_list[i],
                #                                       Image_path, points_005, scores_005, current_points,
                #                                       current_scores,
                #                                       Display_width, Display_height,
                #                                       Image_Ori_W, Image_Ori_H,
                #                                       ssim_t)
        if box_type == "del_all":
            print(f"  {box_type} boxes: {coords}")
            for box in coords:
                img_n, current_count, current_points,current_scores = \
                    interactive_adaptation_boxs_del_all(img_n, box,
                                                        img_path, points_005, scores_005,
                                                        current_points,current_scores,
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

def boxes_F1(log_path,root_path):#用户真实交互，有三种交互，0-2次
    model, device, transform,args = p2p_init_visual_counter()
    ssim_t = args.ssim_t
    maes_ori, mses_ori = [], []
    m_F1point_scores_ori, m_Precision_scores_ori, m_Recall_scores_ori = [], [], []
    maes_cur, mses_cur = [], []
    m_F1point_scores_cur, m_Precision_scores_cur, m_Recall_scores_cur = [], [], []
    # 用来存储所有解析出来的框坐标及对应的文件名
    boxes_list = []
    # 定义正则表达式来匹配文件名和框的坐标（提取元组）
    file_pattern = re.compile(r"IMG_\d+\.tif")
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

                mae_ori,mse_ori,F1point_score_ori,Precision_score_ori,Recall_score_ori,\
                mae_cur,mse_cur,F1point_score_cur,Precision_score_cur,Recall_score_cur=\
                    adaptation_boxes(boxes,model,device,transform,root_path,ssim_t)

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
        log_file.write(f'\nEval Log {time.strftime("%c")}\n')
        log_file.write(f"{args}\n")
        log_file.write(f"m_F1point_score_ori: {m_F1point_score_ori}\n")
        log_file.write(f"m_Precision_score_ori: {m_Precision_score_ori}\n")
        log_file.write(f"m_Recall_score_ori: {m_Recall_score_ori}\n")
        log_file.write(f"mae_ori: {mae_ori}, mse_ori: {mse_ori}\n")

        log_file.write(f"m_F1point_score_cur: {m_F1point_score_cur}\n")
        log_file.write(f"m_Precision_score_cur: {m_Precision_score_cur}\n")
        log_file.write(f"m_Recall_score_cur: {m_Recall_score_cur}\n")
        log_file.write(f"mae_cur: {mae_cur}, mse_cur: {mse_cur}\n")

        log_file.write(f"num of img: {len(interinf)}, ssim_t: {ssim_t}\n")
        # log_file.write(
        #     f"num of del_re_times: {len(del_re_time)}, max_time: {np.max(del_re_time)}, mean_time: {np.mean(del_re_time)}\n")
        # log_file.write(
        #     f"num of add_times: {len(add_time)}, max_time: {np.max(add_time)}, mean_time: {np.mean(add_time)}\n")

def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for training P2PNet', add_help=False)
    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")
    parser.add_argument('--weight_path', default='/mnt/data/zrj/Mymodel/CELLSsplit_v4_best_e20.pth',
                        help='path where the trained weights saved')  # 43090
    # parser.add_argument('--weight_path', default='/mnt/data/zrj/Mymodel/CELLSsplit_v4_best_e1500.pth',
    #                     help='path where the trained weights saved')  # 43090
    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")
    parser.add_argument('--ssim_t', default=0.8, type=int,
                        help="ssim threshold")
    # dataset parameters
    parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for training')

    return parser
def main(args):
    # 与交互式计数方法对比,初始预测模型选择训练20个epoch的
    log_path = "/mnt/data/zrj/prj/AICC/interact_box_log_test192_box33.txt"  # 43090
    root_path = '/mnt/data/zrj/Mydata/CELLSsplit_v4/DATA_ROOT/test'  # 43090
    sixbox_F1(log_path, root_path)

    # #与非交互式计数方法对比,初始预测模型选择训练1500个epoch的
    # log_path = '/mnt/data/zrj/prj/AICC/interact_box_log_test192.txt'
    # root_path = '/mnt/data/zrj/Mydata/CELLSsplit_v4/DATA_ROOT/test'  # 43090
    # boxes_F1(log_path, root_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('P2PNet_simulation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)