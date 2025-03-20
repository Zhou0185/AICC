import argparse
import os
# import pdb
import time
import cv2
import tkinter as tk
import sys
sys.path.append('/mnt/data/zrj/prj/AICC')  # 44
from models import build_model
import torchvision.transforms as standard_transforms
from sklearn.cluster import KMeans
import tkinter.font as font
from tkinter import filedialog, messagebox
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageTk
from scipy.spatial import KDTree

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


class AICCInterface(tk.Tk):
    def __init__(self, path):
        print("__init__")
        super().__init__()
        self.title("Adaptive Interactive Cell Counting Interface")
        self.geometry("1560x1000")
        # self.resizable(False, False)
        self.Main_cv = tk.Canvas(self, width=1460, height=950)
        self.Display_width = 640
        self.Display_height = 640
        # Init Variable
        self.COUNT_RESULT = 0
        self.DRAWING_ENABLED = False
        self.EXEMPLAR_X = None
        self.EXEMPLAR_Y = None
        self.EXEMPLAR_START_X = None
        self.EXEMPLAR_START_Y = None
        self.EXEMPLAR_END_X = None
        self.EXEMPLAR_END_Y = None
        self.Image_path = None
        # Init Layout
        self.init_image_result_area()
        self.init_interactive_counting_area()
        self.init_exemplar_area()
        self.init_menu_bar()
        self.init_visual_counter()
        self.init_popup_menu()
        ##
        self.sizedpoints = []
        self.scores = None
        self.seletedpoints = []
        self.indice = []
        self.indice1 = []
        # pdb.set_trace()
    def init_visual_counter(self):
        print("init_visual_counter")
        if torch.cuda.is_available():
            self.device = "cuda:0"
            print(self.device, '-------')
        else:
            self.device = 'cpu'
            print('cpu-------')

        def get_args_parser():
            parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

            # * Backbone
            parser.add_argument('--backbone', default='vgg16_bn', type=str,
                                help="name of the convolutional backbone to use")
            parser.add_argument('--row', default=2, type=int,
                                help="row number of anchor points")
            parser.add_argument('--line', default=2, type=int,
                                help="line number of anchor points")
            parser.add_argument('--output_dir', default='vis',
                                help='path where to save')
            parser.add_argument('--test_output_dir', default='/mnt/data/zrj/prj/AICC/output',
                                help='path where to save')  # 4_32
            parser.add_argument('--weight_path', default='/mnt/data/zrj/Mymodel/CELLSsplit_v4_best_e20.pth',
                                help='path where the trained weights saved')  # 4_32服务器
            # parser.add_argument('--weight_path', default='/mnt/data/zrj/Mymodel/CELLSsplit_v4_best_e1500.pth',
            #                     help='path where the trained weights saved')  # 4_4服务器

            # CELLSsplitlatest1024b8e2000 CELLSlatestb8e2000 Cellsplitv3lastb4e2000.pth
            # parser.add_argument('--gpu_id', default=0, type=int, help='the gpu used for evaluation')
            parser.add_argument('--gpu_id', default=1, type=int, help='the gpu used for evaluation')

            return parser

        parser = argparse.ArgumentParser('P2PNet evaluation script', parents=[get_args_parser()])
        args = parser.parse_args()
        print(args)
        self.run_log_name = os.path.join(args.test_output_dir, 'interact_log.txt')
        self.run_log_name_add = os.path.join(args.test_output_dir, 'interact_add_log.txt')
        self.run_log_name_del = os.path.join(args.test_output_dir, 'interact_del_log.txt')
        self.run_log_name_box = os.path.join(args.test_output_dir, 'interact_box_log.txt')
        print(self.run_log_name)
        with open(self.run_log_name, "a") as log_file:  # 记录计数用的模型pth等信息
            log_file.write('\nEval Log %s\n' % time.strftime("%c"))
            log_file.write("{}".format(args))
        with open(self.run_log_name_box, "a") as log_file:  # 记录计数用的模型pth等信息
            log_file.write('\nEval Log %s\n' % time.strftime("%c"))
            log_file.write("{}".format(args))
        # device = torch.device('cuda')
        device = torch.device(self.device)
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
        self.visual_counter = model

    def initial_count(self):
        if self.Image_path is None:
            messagebox.showinfo("Image Invalid", "Please select an image.")
            return

        self.indice = []
        self.indice1 = []
        self.similarity_inds = {}
        transform = standard_transforms.Compose([
            standard_transforms.ToTensor(),
            standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        # load the images
        img_raw = Image.open(self.Image_path).convert('RGB')
        # round the size
        width, height = img_raw.size
        new_width = width // 128 * 128
        new_height = height // 128 * 128
        # img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
        img_raw = img_raw.resize((new_width, new_height), Image.LANCZOS)
        # pre-proccessing
        img = transform(img_raw)

        samples = torch.Tensor(img).unsqueeze(0)
        samples = samples.to(self.device)

        # print(samples.shape)
        # run inference
        outputs, self.simifeat = self.visual_counter(samples)  # return out,features_fpn[1]
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]  # [:, :, 1][0]为错误点的概率
        outputs_points = outputs['pred_points'][0]
        self.inter_outputs_scores = outputs_scores
        self.inter_outputs_points = outputs_points

        # self.visual_counter.reset_refinement_module(self.simifeat.shape[-2], self.simifeat.shape[
        #     -1])  # 设置细化模块（空间细化）参数的形状，形状由 self.features 的倒数第二个和最后一个维度确定。n
        self.visual_counter.classification.reset_refinement_module(self.simifeat.shape[-2], self.simifeat.shape[
            -1])  # 设置细化模块（空间细化）参数的形状，形状由 self.features 的倒数第二个和最后一个维度确定。n
        self.visual_counter.to(self.device)
        # threshold = 0.8
        # allpoints = outputs_points[outputs_scores > 0.05].detach().cpu().numpy().tolist()
        self.threshold = 0.5
        # filter the predictions
        points = outputs_points[outputs_scores > self.threshold].detach().cpu().numpy().tolist()
        self.indice = [i for i in range(len(outputs_scores)) if outputs_scores[i] > self.threshold]
        print(self.indice)  # 初始预测正确点的下标
        predict_cnt = len(points)
        print(predict_cnt)

        # self.scores = outputs_scores
        # root = tkinter.Tk()
        def SizedPoints(points):
            sizedpoints = []
            data = points
            llen = len(data)
            for i in range(llen):
                x = round(int(data[i][0]) * self.Display_width / self.Image_Ori_W)
                y = round(int(data[i][1]) * self.Display_height / self.Image_Ori_H)
                sizedpoints.append((x, y))
                # print(int(data[i][0]),int(data[i][1]),"__",x,y)
            return sizedpoints

        self.sizedpoints = SizedPoints(points)  ##Qview_0
        print("self.sizedpoints:", len(self.sizedpoints))
        self.count_res_string_var.set(str(predict_cnt))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('\n{}'.format(self.Image_path))
            log_file.write('{}{}'.format(" initial_count:", predict_cnt))
        with open(self.run_log_name_box, "a") as log_file:
            log_file.write('\n{}'.format(self.Image_path))
            log_file.write('{}{}'.format(" initial_count:", predict_cnt))
        self.initial_num = predict_cnt
        self.current_num = predict_cnt

        def DrawfPoints(points, Img_path):
            data = points
            llen = len(data)
            img = cv2.imread(Img_path)
            # noname = Img_path.split("/")[-1]
            noname = os.path.basename(Img_path).split(".")[0]
            dot_size = int(16 / 4096 * img.shape[0])
            # print(img.shape[0])
            print("dot_size:", dot_size)
            for i in range(llen):
                # print(data[i])
                x = int(data[i][0])
                y = int(data[i][1])
                tempmask = [x, y]
                # print(tempmask)
                # img = cv2.circle(img, tempmask, dot_size, (0, 255, 255), -1)
                img = cv2.circle(img, tempmask, dot_size, (255, 0, 255), -1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # plt.imshow(img)
            # plt.title(noname + " pd:" + str(llen))
            # plt.show()  # 显示图片
            return img

        imgPointed = DrawfPoints(points, self.Image_path)
        imgPointed = Image.fromarray(np.uint8(imgPointed))  # fromarray OpenCV转换成PIL.Image格式
        imgPointed = imgPointed.resize((self.Display_width, self.Display_height))
        self.output_image = imgPointed.copy()  # 放在ImageTk.PhotoImage(imgPointed)之前
        self.output_image_back_up = imgPointed.copy()
        imgPointed.save("./{}_init.tif".format(self.Image_name))
        imgPointed = ImageTk.PhotoImage(imgPointed)
        self.Visual_Label.configure(image=imgPointed)
        self.Visual_Label.image = imgPointed
        #imgPointed.save("./{}_init.tif".format(self.Image_name))

        if self.Gt_path:
            with open(self.Gt_path, "r") as f:
                dots = f.readlines()
            gt_cnt = len(dots)
            print("gt_cnt", gt_cnt)
            with open(self.run_log_name, "a") as log_file:
                log_file.write('{}{}'.format(" gt_count:", gt_cnt))
            with open(self.run_log_name_box, "a") as log_file:
                log_file.write('{}{}'.format(" gt_count:", gt_cnt))

    def init_popup_menu(self):
        print("init_popup_menu")
        # For range speification
        self.popup_menu = tk.Menu(self, tearoff=0)
        self.popup_menu.add_command(label='add missing point', command=self.popup_menu_2)
        self.popup_menu.add_command(label='remove redundant points', command=self.popup_menu_1)
        self.popup_menu.add_command(label='remove erroneous points', command=self.popup_menu_0)

        # self.popup_menu.add_command(label='box_add', command=self.popup_menu_3)
        # self.popup_menu.add_command(label='box_del', command=self.popup_menu_4)

    def init_exemplar_area(self):
        print("init_exemplar_area")
        RECTANGLE_BASE_X = 820
        RECTANGLE_BASE_Y = 740
        BASE_X = 900
        BASE_Y = 740

        self.Main_cv.create_rectangle(RECTANGLE_BASE_X, RECTANGLE_BASE_Y, RECTANGLE_BASE_X + 600,
                                      RECTANGLE_BASE_Y + 150)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Interaction Providing", fg='black')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x=BASE_X + 25, y=BASE_Y - 12)

        self.ExempalrB = tk.Button(self, text="DrawExempalr", command=self.activate_drawing)
        self.ExempalrB['font'] = font.Font(size=15)
        self.ExempalrB.place(x=BASE_X + 0, y=BASE_Y + 70)
        self.ExempalrB["width"] = 15
        self.ExempalrB["height"] = 1
        self.ExempalrB["relief"] = tk.GROOVE

        # self.ExempalrUnDoB = tk.Button(self, text="Undo", command=self.exemplar_undo)
        # self.ExempalrUnDoB['font'] = font.Font(size=15)
        # self.ExempalrUnDoB.place(x=BASE_X + 300, y=BASE_Y + 70)
        # self.ExempalrUnDoB["width"] = 15
        # self.ExempalrUnDoB["height"] = 1
        # self.ExempalrUnDoB["relief"] = tk.GROOVE

        self.ExempalrResetB = tk.Button(self, text="Reset", command=self.exemplar_reset)
        self.ExempalrResetB['font'] = font.Font(size=15)
        self.ExempalrResetB.place(x=BASE_X + 300, y=BASE_Y + 70)
        self.ExempalrResetB["width"] = 15
        self.ExempalrResetB["height"] = 1
        self.ExempalrResetB["relief"] = tk.GROOVE

        self.EXEMPLAR_LIST = []

    def init_interactive_counting_area(self):
        print("init_interactive_counting_area")
        RECTANGLE_BASE_X = 10
        RECTANGLE_BASE_Y = 740
        BASE_X = 100
        BASE_Y = 740

        # Init Layout
        self.Main_cv.create_rectangle(RECTANGLE_BASE_X, RECTANGLE_BASE_Y, RECTANGLE_BASE_X + 600,
                                      RECTANGLE_BASE_Y + 150)
        self.Main_cv.pack()

        # Count Label
        count_label = tk.Label(self, text="Counting Result: ", fg='blue')
        count_label.pack()
        count_label['font'] = font.Font(size=18)
        count_label.place(x=BASE_X + 0, y=BASE_Y + 20)
        self.count_res_string_var = tk.StringVar()
        self.count_res_string_var.set("0")
        self.count_res_label = tk.Label(self, textvariable=self.count_res_string_var, fg='blue')
        self.count_res_label['font'] = font.Font(size=18)
        self.count_res_label.pack()
        self.count_res_label.place(x=BASE_X + 300, y=BASE_Y + 21)

        self.count_button = tk.Button(self, text="Initial Count", command=self.initial_count)
        self.count_button['font'] = font.Font(size=15)
        self.count_button.place(x=BASE_X + 0, y=BASE_Y + 70)
        self.count_button["width"] = 15
        self.count_button["height"] = 1
        self.count_button["relief"] = tk.GROOVE

    def init_image_result_area(self):
        print("init_image__result_area")
        # Init Image Area
        IMAGE_RECTANGLE_BASE_X = 20
        IMAGE_RECTANGLE_BASE_Y = 30


        # self.Main_cv.create_rectangle(IMAGE_RECTANGLE_BASE_X, IMAGE_RECTANGLE_BASE_Y, IMAGE_RECTANGLE_BASE_X + 700, IMAGE_RECTANGLE_BASE_Y + 420)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Input Image", fg='black')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x=IMAGE_RECTANGLE_BASE_X + 25, y=IMAGE_RECTANGLE_BASE_Y - 12)
        self.Input_Image_Label = tk.Label(self, text="")
        self.Input_Image_Label.pack()
        self.Input_Image_Label.place(x=IMAGE_RECTANGLE_BASE_X + 25, y=IMAGE_RECTANGLE_BASE_Y + 28)
        self.EXEMPLAR_LIST = []

        default_img = Image.open('interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Input_Image = default_img
        self.Input_Image_backup = self.Input_Image.copy()
        self.Input_Image_with_exemplar = self.Input_Image.copy()

        self.Input_Image_Label.configure(image=default_show)
        self.Input_Image_Label.image = default_show

        RESULT_RECTANGLE_BASE_X = 840
        RESULT_RECTANGLE_BASE_Y = 30
        # self.Main_cv.create_rectangle(RESULT_RECTANGLE_BASE_X, RESULT_RECTANGLE_BASE_Y, RESULT_RECTANGLE_BASE_X + 700, RESULT_RECTANGLE_BASE_Y + 420)
        self.Main_cv.pack()
        Input_Image_Text = tk.Label(self, text="Visualization", fg='black')
        Input_Image_Text.pack()
        Input_Image_Text['font'] = font.Font(size=15)
        Input_Image_Text.place(x=RESULT_RECTANGLE_BASE_X + 25, y=RESULT_RECTANGLE_BASE_Y - 12)
        self.Visual_Label = tk.Label(self, text="")
        self.Visual_Label.pack()
        self.Visual_Label.place(x=RESULT_RECTANGLE_BASE_X + 25, y=RESULT_RECTANGLE_BASE_Y + 28)

        default_img = Image.open('interface_default.jpg')
        default_img = default_img.resize((self.Display_width, self.Display_height))
        default_show = ImageTk.PhotoImage(default_img)
        self.Visual_Label.configure(image=default_show)
        self.Visual_Label.image = default_show
        # self.Visual_Label.bind("<Button-1>", self.finding_points_del)
        # self.Visual_Label.bind("<Button-2>", self.finding_points_add)
        self.Visual_Label.bind('<Button-3>', self.popup)
        self.Visual_Label.bind("<Button-1>", self.start_draw_rectangle)
        self.Visual_Label.bind("<B1-Motion>", self.update_rectangle)
        self.Visual_Label.bind("<ButtonRelease-1>", self.end_draw_rectangle)

    def init_menu_bar(self):
        print("init_menu_bar")
        # Init menu bar
        self.MB = tk.Menu(self)
        self.config(menu=self.MB)
        self.MB.add_command(label="Load Image", command=self.Menu_Bar_Load_Image)

    def Menu_Bar_Load_Image(self):
        # Init Variable
        print('Load_Image--------------------------------')
        self.COUNT_RESULT = 0
        self.DRAWING_ENABLED = False
        self.EXEMPLAR_START_X = None
        self.EXEMPLAR_START_Y = None
        self.EXEMPLAR_END_X = None
        self.EXEMPLAR_END_Y = None
        self.Image_path = None
        self.Gt_path = None
        # Init Layout
        self.init_image_result_area()
        self.init_interactive_counting_area()
        self.init_exemplar_area()
        self.init_menu_bar()
        # self.init_visual_counter()
        self.init_popup_menu()
        # The only important para
        # self.Image_path = filedialog.askopenfilename(initialdir="./", title="Select image.")
        self.Image_path = filedialog.askopenfilename(
            initialdir="/mnt/disk3/zrj/MYP2PNET_ROOT/crowd_datasets/CELLSsplit_v4/DATA_ROOT/test/images",
            title="Select image.")
        img_open = Image.open(self.Image_path)
        self.Image_name = self.Image_path.split("/")[-1].split(".")[0]
        gt_path = self.Image_path.replace("images", "test_file").replace(".tif", ".txt")
        print(gt_path)
        if os.path.exists(gt_path):
            self.Gt_path = gt_path
        self.max_hw = 1504
        W, H = img_open.size
        img_open = img_open.resize((self.Display_width, self.Display_height))
        img_show = ImageTk.PhotoImage(img_open)
        self.Input_Image_Label.configure(image=img_show)
        self.Input_Image_Label.image = img_show
        self.Input_Image = img_open.copy()
        self.Input_Image_with_exemplar = img_open.copy()
        self.Input_Image_backup = img_open.copy()
        self.Image_Ori_W = W
        self.Image_Ori_H = H


    def activate_drawing(self):
        self.DRAWING_ENABLED = True

    def start_draw_rectangle(self, event):
        if not self.DRAWING_ENABLED:
            return
        self.EXEMPLAR_START_X = event.x
        self.EXEMPLAR_START_Y = event.y
        self.EXEMPLAR_BBX = (self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y)

    def update_rectangle(self, event):
        if not self.DRAWING_ENABLED or self.EXEMPLAR_BBX is None or self.output_image is None:
            return
        self.EXEMPLAR_BBX = (self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, event.x, event.y)

        # self.Input_Image_with_exemplar = self.Input_Image.copy()
        # draw = ImageDraw.Draw(self.Input_Image_with_exemplar)
        # draw.rectangle(self.EXEMPLAR_BBX, outline='red', width=2)
        # photo = ImageTk.PhotoImage(self.Input_Image_with_exemplar)
        # self.Input_Image_Label.configure(image=photo)
        # self.Input_Image_Label.image = photo

        self.Input_Image_with_exemplar = self.output_image.copy()
        draw = ImageDraw.Draw(self.Input_Image_with_exemplar)
        draw.rectangle(self.EXEMPLAR_BBX, outline='red', width=2)
        photo = ImageTk.PhotoImage(self.Input_Image_with_exemplar)
        self.Visual_Label.configure(image=photo)
        self.Visual_Label.image = photo

    def end_draw_rectangle(self, event):
        if not self.DRAWING_ENABLED:
            return

        # self.Input_Image = self.Input_Image_with_exemplar.copy()
        self.Visual_Label.image = self.Input_Image_with_exemplar.copy()
        photo = ImageTk.PhotoImage(self.Input_Image_with_exemplar.copy())
        self.Visual_Label.configure(image=photo)
        self.Visual_Label.image = photo
        self.output_image = self.Input_Image_with_exemplar.copy()
        self.EXEMPLAR_END_X = event.x
        self.EXEMPLAR_END_Y = event.y
        x_min = min(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        x_max = max(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        y_min = min(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        y_max = max(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        x_len = x_max - x_min
        y_len = y_max - y_min
        self.EXEMPLAR_LIST.append((x_min, y_min, x_max, y_max))
        self.EXEMPLAR_BBX = None
        self.DRAWING_ENABLED = False
        print(self.EXEMPLAR_LIST)

    def similarity_fun(self, point, tree):
        return

    def finding_points_del(self, event):
        if self.DRAWING_ENABLED:
            self.EXEMPLAR_START_X = event.x
            self.EXEMPLAR_START_Y = event.y
            self.EXEMPLAR_BBX = (
            self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y)
            return
        if not self.DRAWING_ENABLED:
            # self.EXEMPLAR_BBX = (self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, event.x, event.y)
            tempoint = np.array([event.x, event.y])
            print(tempoint, "----点击的坐标")
            # print("self.sizedpoints:", type(self.sizedpoints))
            # self.sizedpoints = SizedPoints(points)
            if (self.sizedpoints != None):
                tree = KDTree(self.sizedpoints)  # 全部预测的点构建树
                _, closest_point_index = tree.query(tempoint)  # 点击位置最近点的下标
                closest_point = self.sizedpoints[closest_point_index]  # 得到点击的点坐标
                print("离最近的点：", closest_point, " 下标：", closest_point_index, "/", len(self.sizedpoints))
                self.indice1.append(closest_point_index)  # 加入待删除点列表
                self.Visual_image_finding = self.output_image.copy()  # 画图标记
                draw = ImageDraw.Draw(self.Visual_image_finding)
                draw.ellipse((closest_point[0] - 2, closest_point[1] - 2, closest_point[0] + 2, closest_point[1] + 2),
                             width=2,
                             outline='red', fill=(255, 0, 0))
                self.Visual_Label.image = self.Visual_image_finding.copy()
                photo = ImageTk.PhotoImage(self.Visual_image_finding.copy())
                self.Visual_Label.configure(image=photo)
                self.Visual_Label.image = photo
                self.output_image = self.Visual_image_finding.copy()
                self.seletedpoints.append(closest_point)  # 加入已选择点列表
                print("len(self.seletedpoints):", len(self.seletedpoints))
                # self.inter_outputs_points
                ind = tree.query_ball_point(closest_point, r=15)  # 点击点的附近点坐标
                ind = [x for x in ind if x != closest_point_index]
                print("点击点的附近点个数，坐标")
                print(len(ind))
                for i in ind:
                    print(self.sizedpoints[int(i)])
                # self.similarity_inds["pointnc:{},{}".format(closest_point[0], closest_point[1])] = len(ind)
                self.similarity_inds["pointnc_ind:{}".format(closest_point_index)] = len(ind)
                print(self.similarity_inds)

    def finding_points_add(self, event):
        # self.EXEMPLAR_BBX = (self.EXEMPLAR_START_X, self.EXEMPLAR_START_Y, event.x, event.y)
        tempoint = np.array([event.x, event.y])
        print(tempoint, "----++点击的坐标")
        # print("self.sizedpoints:", type(self.sizedpoints))
        # self.sizedpoints = SizedPoints(points)
        if (self.sizedpoints != None):
            tree = KDTree(self.sizedpoints)
            _, closest_point_index = tree.query(tempoint)
            closest_point = self.sizedpoints[closest_point_index]
            print("离最近的点：", closest_point, " 下标：", closest_point_index, "/", len(self.sizedpoints))
            self.indice1.append(closest_point_index)
            self.Visual_image_finding = self.output_image.copy()
            draw = ImageDraw.Draw(self.Visual_image_finding)
            draw.ellipse((closest_point[0] - 2, closest_point[1] - 2, closest_point[0] + 2, closest_point[1] + 2),
                         width=2,
                         outline='red', fill=(255, 0, 255))
            self.Visual_Label.image = self.Visual_image_finding.copy()
            photo = ImageTk.PhotoImage(self.Visual_image_finding.copy())
            self.Visual_Label.configure(image=photo)
            self.Visual_Label.image = photo
            self.output_image = self.Visual_image_finding.copy()
            self.seletedpoints.append(closest_point)
            print("len(self.seletedpoints):", len(self.seletedpoints))
            # self.inter_outputs_points
            width, height = self.output_image.size
            row_no = 0
            col_no = 0
            split_num = 4
            split_lenw = width // split_num
            split_lenh = height // split_num
            for i in range(16):
                if (closest_point[0] > row_no * split_lenw and closest_point[0] < (row_no + 1) * (split_lenw)
                        and closest_point[1] > col_no * split_lenh and closest_point[1] < (col_no + 1) * (split_lenh)):
                    print("属于区域：", row_no, col_no)
                    # crop_area = (row_no * split_len, col_no * split_len, (row_no + 1) * (split_len), (col_no + 1) * (split_len))
                if (col_no % split_num == split_num - 1):
                    col_no = 0
                    row_no = row_no + 1
                else:
                    col_no = col_no + 1
            # round(int(closest_point[0])
            # round(int(closest_point[1])

    def exemplar_reset(self):
        print("exemplar_reset")


    def popup_menu_0(self):
        print("popup_menu_0")
        # self.estimate_lb = -1
        # self.estimate_ub = 0
        # self.interactive_adaptation()
        #
        # with open(self.run_log_name_box, "a") as log_file:
        #     log_file.write('\n{}#sized_box_add:{}\n'.format(self.Image_path,self.EXEMPLAR_LIST))
        with open(self.run_log_name, "a") as log_file:
            log_file.write(' del_all_action ')
        self.interactive_adaptation_box_del_all()

    def popup_menu_1(self):
        print("popup_menu_1")
        with open(self.run_log_name, "a") as log_file:
            log_file.write(' del_re_action ')
        #
        # with open(self.run_log_name_del, "a") as log_file:
        #     log_file.write('\n{}\n'.format(self.Image_path))
        # if self.Gt_path:
        #     with open(self.Gt_path, "r") as f:
        #         dots = f.readlines()
        #     gt_cnt = len(dots)
        #     with open(self.run_log_name_del, "a") as log_file:
        #         log_file.write('{}{}'.format("#gt_count:", gt_cnt))
        self.interactive_adaptation_box_del_re()

    def popup_menu_2(self):
        print("popup_menu_2")
        with open(self.run_log_name, "a") as log_file:
            log_file.write(' add_action ')
        # with open(self.run_log_name_add, "a") as log_file:
        #     log_file.write('\n{}\n'.format(self.Image_path))
        # if self.Gt_path:
        #     with open(self.Gt_path, "r") as f:
        #         dots = f.readlines()
        #     gt_cnt = len(dots)
        #     with open(self.run_log_name_add, "a") as log_file:
        #         log_file.write('{}{}'.format("#gt_count:", gt_cnt))
        self.interactive_adaptation_box_add()

    def popup_menu_3(self):
        print("popup_menu_3")
        # self.estimate_lb = -1
        # self.estimate_ub = 0
        # self.interactive_adaptation()
        with open(self.run_log_name_box, "a") as log_file:
            log_file.write('\n{}#sized_box_add:{}'.format(self.Image_path, self.EXEMPLAR_LIST))
        self.EXEMPLAR_LIST = []

    def popup_menu_4(self):
        print("popup_menu_4")
        # self.estimate_lb = -1
        # self.estimate_ub = 0
        # self.interactive_adaptation()
        with open(self.run_log_name_box, "a") as log_file:
            log_file.write('#sized_box_del:{}'.format(self.EXEMPLAR_LIST))
        self.EXEMPLAR_LIST = []

    def popup(self, event):
        global click_index
        # if self.label is not None:
        #     # Convert to Image coordinate
        #     trans_y = int((self.Real_Height / self.Display_height) * event.y)
        #     trans_x = int((self.Real_Width / self.Display_width) * event.x)
        #     #self.selected_region = self.label[trans_y, trans_x]
        click_index = 0
        self.popup_menu.post(event.x_root, event.y_root)

    # 1 得到用户框选的图块
    # 2 遍历目前的点（已经画在图上的self.sizedpoints）
    # 3 以目前点i为中心的框，计算框内的点个数
    # 4 若个数大于1，计算ssim，小于0.5则continue
    # 5 ssim大于0.5，则只保留框内的第一个点且记入selected，其余删除
    def interactive_adaptation_box_del_re(self):
        def get_area_points(x_len, y_len, points_view, point, selectedp_list):
            # pointsraw = torch.tensor(self.sizedpoints).view(-1, 2).tolist()#只在画框内的预测点
            pnum = 0
            plist = []
            for point1 in points_view:
                if int(point1[0]) > point[0] - x_len // 2 and int(point1[0]) < point[0] + x_len // 2 \
                        and int(point1[1]) > point[1] - y_len // 2 and int(point1[1]) < point[1] + y_len // 2 \
                        and point1 not in selectedp_list:
                    # print(point1)
                    pnum = pnum + 1
                    plist.append(point1)
            return pnum, plist

        start = time.time()
        del_points = []
        x_min = min(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        x_max = max(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        y_min = min(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        y_max = max(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        x_len = x_max - x_min
        y_len = y_max - y_min
        # self.EXEMPLAR_LIST.append((x_min, y_min, x_max, y_max))
        # 得到框内的图像信息
        crop_area_base = (x_min, y_min, x_max, y_max)
        crop_img_base_rgb = self.Input_Image.copy().crop(crop_area_base).convert('RGB')
        # plt.imshow(crop_img_base_rgb)
        # plt.show()  # 显示图片
        # 为所有_预测点_画框
        selectedp_list = []
        for point in self.sizedpoints[:]:  # 遍历在新的列表操作，删除是在原来的列表操作 Qview_t
            points_view = self.sizedpoints.copy()  # 只在画框内的预测点
            pnum, plist = get_area_points(x_len, y_len, points_view, point, selectedp_list)
            if pnum > 1:
                #print("删除点，并更新已删除点，后面作为判断，防止重复删除")
                n_x_min = point[0] - x_len // 2;
                n_x_max = point[0] + x_len // 2
                n_y_min = point[1] - y_len // 2;
                n_y_max = point[1] + y_len // 2
                crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
                crop_img = self.Input_Image.copy().crop(crop_area)
                # crop_img = crop_img.convert('L')  # 转换成灰度图
                crop_img = crop_img.convert('RGB')
                # plt.imshow(crop_img)
                # plt.show()  # 显示图片
                if crop_img_base_rgb.size != crop_img.size:
                    crop_img = crop_img.resize((crop_img_base_rgb.size[0], crop_img_base_rgb.size[1]), Image.LANCZOS)
                s_score = ssim(crop_img, crop_img_base_rgb)
                if s_score < 0.8:
                    #print("图像ssim相似度:", s_score, " 跳过")
                    continue
                #print("图像ssim相似度:", s_score, "不跳过")

                selectedp_list.append(plist[0])  # 标记第一个点，防止后续重复删除
                for i in range(len(plist) - 1):
                    point2 = plist[i + 1]  # 只保留第一个点
                    self.sizedpoints.remove(point2)
                    del_points.append(point2)

                    self.Visual_image_finding = self.output_image.copy()  # 画图标记
                    draw = ImageDraw.Draw(self.Visual_image_finding)
                    draw.ellipse((point2[0] - 2, point2[1] - 2, point2[0] + 2, point2[1] + 2), width=1,
                                 outline='black', fill=(0, 0, 0))
                    # draw.ellipse((point2[0] - 4, point2[1] - 4, point2[0] + 4, point2[1] + 4), width=1,
                    #              outline='black', fill=None)
                    self.Visual_Label.image = self.Visual_image_finding.copy()
                    photo = ImageTk.PhotoImage(self.Visual_image_finding.copy())
                    self.Visual_Label.configure(image=photo)
                    self.Visual_Label.image = photo
                    self.output_image = self.Visual_image_finding.copy()
                print('self.sizedpoints-：', len(self.sizedpoints))
                self.count_res_string_var.set("0")
                self.count_res_string_var.set(str(len(self.sizedpoints)))
        end = time.time()
        print("adapative time:", end - start)
        self.output_image.save("../output/{}_del_re.tif".format(self.Image_name))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('#current_cnt:{}'.format(len(self.sizedpoints)))
            log_file.write('#del_re_point:{}'.format(del_points))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('#sized_interact_box_area:{}'.format(crop_area_base))
        with open(self.run_log_name_box, "a") as log_file:
            log_file.write('#sized_box_del_re:{}'.format(crop_area_base))
            log_file.write('#current_cnt_del_re:{}'.format(len(self.sizedpoints)))
        # with open(self.run_log_name_del, "a") as log_file:
        #     log_file.write('#current_cnt:{}'.format(len(self.sizedpoints)))

    def interactive_adaptation_box_del_all(self):
        del_points = []
        x_min = min(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        x_max = max(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        y_min = min(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        y_max = max(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        x_len = x_max - x_min
        y_len = y_max - y_min
        # self.EXEMPLAR_LIST.append((x_min, y_min, x_max, y_max))
        # 得到框内的图像信息
        crop_area_base = (x_min, y_min, x_max, y_max)
        # crop_img_base = self.Input_Image.copy().crop(crop_area_base)
        crop_img_base_rgb = self.Input_Image.copy().crop(crop_area_base).convert('RGB')
        # crop_img_base = crop_img_base_rgb.convert('L')#转换成灰度图
        # plt.imshow(crop_img_base_rgb)
        # plt.show()  # 显示图片
        # 为所有_预测点_画框
        selectedp_list = []
        # for point in self.sizedpoints:
        start = time.time()
        for point in self.sizedpoints[:]:  # 遍历在新的列表操作，删除是在原来的列表操作
            # pointsraw = self.sizedpoints.copy()#只在画框内的预测点
            # pnum,plist = get_area_points(x_len, y_len, pointsraw,point,selectedp_list)

            print("删除点，并更新已删除点，后面作为判断，防止重复删除")
            n_x_min = point[0] - x_len // 2;
            n_x_max = point[0] + x_len // 2
            n_y_min = point[1] - y_len // 2;
            n_y_max = point[1] + y_len // 2
            # Ensure the crop area is within the image boundaries (0 to 639 for 640x640 image)
            n_x_min = max(0, n_x_min)  # Ensure n_x_min is at least 0
            n_x_max = min(639, n_x_max)  # Ensure n_x_max is at most 639
            n_y_min = max(0, n_y_min)  # Ensure n_y_min is at least 0
            n_y_max = min(639, n_y_max)  # Ensure n_y_max is at most 639
            crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
            crop_img = self.Input_Image.copy().crop(crop_area)
            # crop_img = crop_img.convert('L')  # 转换成灰度图
            crop_img = crop_img.convert('RGB')
            # plt.imshow(crop_img)
            # plt.show()  # 显示图片
            if crop_img_base_rgb.size != crop_img.size:
                crop_img = crop_img.resize((crop_img_base_rgb.size[0], crop_img_base_rgb.size[1]), Image.LANCZOS)
            s_score = ssim(crop_img, crop_img_base_rgb)
            if s_score < 0.8:
                print("图像ssim相似度:", s_score, " 跳过")
                continue
            print("图像ssim相似度:", s_score, "不跳过")
            self.sizedpoints.remove(point)
            del_points.append(point)

            self.Visual_image_finding = self.output_image.copy()  # 画图标记
            draw = ImageDraw.Draw(self.Visual_image_finding)
            draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), width=1,
                         outline='black', fill=(0, 0, 0))
            # draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), width=1,
            #              outline='black', fill=None)

            self.Visual_Label.image = self.Visual_image_finding.copy()
            photo = ImageTk.PhotoImage(self.Visual_image_finding.copy())
            self.Visual_Label.configure(image=photo)
            self.Visual_Label.image = photo
            self.output_image = self.Visual_image_finding.copy()
            print('self.sizedpoints_-：', len(self.sizedpoints))
            self.count_res_string_var.set("0")
            self.count_res_string_var.set(str(len(self.sizedpoints)))
        end = time.time()
        print("adapative time:", end - start)
        self.output_image.save("../output/{}_del_all.tif".format(self.Image_name))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('#current_cnt:{}'.format(len(self.sizedpoints)))
            log_file.write('#del_all_point:{}'.format(del_points))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('#sized_interact_box_area:{}'.format(crop_area_base))
        with open(self.run_log_name_box, "a") as log_file:
            log_file.write('#sized_box_del_all:{}'.format(crop_area_base))
            log_file.write('#current_cnt_del_all:{}'.format(len(self.sizedpoints)))
        # with open(self.run_log_name_del, "a") as log_file:
        #     log_file.write('#current_cnt:{}'.format(len(self.sizedpoints)))

    # 1 得到用户框选的图块
    # 2 遍历阈值为0.05的点（覆盖了正确点的但会有许多重复的点）
    # 3 点筛选：
    #       1判断点是否已被selected，是则continune
    #       2比较当前点的灰度值与框选图块最大灰度值，小于一半则continue（视为背景的点）
    #       3#计算以点i为中心，画框范围内的的目前点（阈值0.5的点）个数，大于1则continue（说明该物体已被计数，不用重复计数）
    #           1若个数为0，计算ssim，小于0.5则continue
    #           2ssim>0.5，选择点i,并把框内所有 点加入队列selected，防止重复
    def interactive_adaptation_box_add(self):
        def SizedPoints(points):
            sizedpoints = []
            data = points
            llen = len(data)
            for i in range(llen):
                x = round(int(data[i][0]) * self.Display_width // self.Image_Ori_W)
                y = round(int(data[i][1]) * self.Display_height // self.Image_Ori_H)
                sizedpoints.append((x, y))
                # print(int(data[i][0]),int(data[i][1]),"__",x,y)
            return sizedpoints

        def DrawPointi(pointsraw, sindex):
            self.Visual_image_finding = self.output_image.copy()  # 画图标记
            draw = ImageDraw.Draw(self.Visual_image_finding)
            draw.ellipse((pointsraw[sindex][0] - 2, pointsraw[sindex][1] - 2, pointsraw[sindex][0] + 2,
                          pointsraw[sindex][1] + 2), width=1,
                         outline='red', fill=(255, 0, 0))
            # draw.ellipse((pointsraw[sindex][0] - 4, pointsraw[sindex][1] - 4, pointsraw[sindex][0] + 4,
            #               pointsraw[sindex][1] + 4), width=1, outline='red', fill=None)
            self.Visual_Label.image = self.Visual_image_finding.copy()
            photo = ImageTk.PhotoImage(self.Visual_image_finding.copy())
            self.Visual_Label.configure(image=photo)
            self.Visual_Label.image = photo
            self.output_image = self.Visual_image_finding.copy()

        # Initial setup for bounding box
        add_points = []
        x_min = min(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        x_max = max(self.EXEMPLAR_START_X, self.EXEMPLAR_END_X)
        y_min = min(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        y_max = max(self.EXEMPLAR_START_Y, self.EXEMPLAR_END_Y)
        x_len = x_max - x_min
        y_len = y_max - y_min

        # 得到框内的图像信息
        crop_area_base = (x_min, y_min, x_max, y_max)
        crop_img_base_rgb = self.Input_Image.copy().crop(crop_area_base).convert('RGB')
        # crop_img_base = crop_img_base_rgb.convert('L')#转换成灰度图
        # plt.imshow(crop_img_base_rgb)
        # plt.show()  # 显示图片

        # rgb
        # 将图像转换为 numpy 数组并展平
        crop_img_base_rgb_array = np.array(crop_img_base_rgb)
        # 将图像展平成 (num_pixels, 3) 的二维数组，每行代表一个像素的 RGB 值
        pixels_rgb = crop_img_base_rgb_array.reshape(-1, 3)
        # 使用 K-means 进行聚类，将图像分为两类（细胞和背景）
        kmeans_rgb = KMeans(n_clusters=2, random_state=0).fit(pixels_rgb)
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
        # background_pixels = crop_img_base_rgb_array[mask == 0]
        # 计算细胞部分和背景部分的平均像素值
        average_cell_pixel = np.mean(cell_pixels.mean(axis=0))
        # average_background_pixel = background_pixels.mean(axis=0)

        # 为所有点画框(置信度大于0.05)
        selectedp_list = []
        sizedpointsraw = SizedPoints(self.inter_outputs_points[self.inter_outputs_scores > 0.05])  # Q_0.05
        points_005 = torch.tensor(sizedpointsraw).view(-1, 2).tolist()  # 画框内的全部锚点
        pointsraw1 = torch.tensor(sizedpointsraw).view(-1, 2).tolist()  # 画框内的全部锚点，用于计计算框最中间的点

        # Open and resize the input image for pixel comparison
        # img_raw = Image.open(self.Image_path).convert('L').resize((self.Display_width, self.Display_height))
        img_raw = Image.open(self.Image_path).convert('RGB').resize((self.Display_width, self.Display_height))
        img_array = np.array(img_raw)

        #start = time.time()
        # 为所有点遍历(置信度大于0.05)
        for i in range(len(points_005)):  # Q_0.05
            pnum = 0
            if points_005[i] in selectedp_list:
                print("point selected,continue")
                continue

            # Ensure points stay within image bounds
            if (points_005[i][0] >= 640):
                points_005[i][0] = 639
            if (points_005[i][1] >= 640):
                points_005[i][1] = 639
            print(points_005[i][0], points_005[i][1])
            currentpoint_pixel = np.mean(img_array[points_005[i][1]][points_005[i][0]])  # ---注意翻一下横纵坐标
            #print("该点的像素值:", currentpoint_pixel, " 目标参考像素值average_cell_pixel：", average_cell_pixel)
            if (abs(currentpoint_pixel - average_cell_pixel) > 20):
                print("point's pixel likes background,continue ", "currentpoint_pixel:", currentpoint_pixel,
                      "average_cell_pixel：", average_cell_pixel)
                continue

            def get_area_points(x_len, y_len, points_view, point, selectedp_list):
                # pointsraw = torch.tensor(self.sizedpoints).view(-1, 2).tolist()#只在画框内的预测点
                pnum = 0
                plist = []
                for point1 in points_view:
                    if point[0] - x_len // 2 < int(point1[0]) < point[0] + x_len // 2 \
                            and point[1] - y_len // 2 < int(point1[1]) < point[1] + y_len // 2 \
                            and point1 not in selectedp_list:
                        # print(point1)
                        pnum = pnum + 1
                        plist.append(point1)
                return pnum, plist

            points_view = self.sizedpoints.copy()  # 当前预测点Qview_t
            pnum, _ = get_area_points(x_len, y_len, points_view, points_005[i],selectedp_list)

            if pnum == 0:
                print("该框中无预测点，选择目前点为预测点（以目前点画的框，已在框的中间）")
                n_x_min = points_005[i][0] - x_len // 2;
                n_x_max = points_005[i][0] + x_len // 2
                n_y_min = points_005[i][1] - y_len // 2;
                n_y_max = points_005[i][1] + y_len // 2
                crop_area = (n_x_min, n_y_min, n_x_max, n_y_max)
                crop_img = self.Input_Image.copy().crop(crop_area)
                crop_img = crop_img.convert('RGB')  # 转换成灰度图
                if crop_img_base_rgb.size != crop_img.size:
                    # crop_img = crop_img.resize((crop_img_base.size[0], crop_img_base.size[1]), Image.LANCZOS)
                    crop_img = crop_img.resize((crop_img_base_rgb.size[0], crop_img_base_rgb.size[1]), Image.LANCZOS)
                print(crop_img.size, crop_img_base_rgb.size)
                s_score = ssim(crop_img, crop_img_base_rgb)
                # plt.imshow(crop_img)
                # plt.show()  # 显示图片
                if s_score < 0.8:
                    print("ssim:", s_score, " continue")
                    continue
                print("ssim:", s_score, "not continue")

                # 选择框内最中间的点
                for j in range(len(pointsraw1)):  # 把 框内所有0.05点加入队列，防止重复 但加点操作不需要？
                    if int(pointsraw1[j][0]) > n_x_min and int(pointsraw1[j][0]) < n_x_max \
                            and int(pointsraw1[j][1]) > n_y_min and int(pointsraw1[j][1]) < n_y_max:
                        # and pointsraw[i] not in selectedp_list:
                        selectedp_list.append(pointsraw1[j])
                self.sizedpoints.append(points_005[i])  # Qview_t
                add_points.append(points_005[i])
                # print(pointsraw1[sindex],'==：',pointsraw[i])
                print('self.sizedpoints_+：', len(self.sizedpoints))
                self.count_res_string_var.set("0")
                self.count_res_string_var.set(str(len(self.sizedpoints)))
                DrawPointi(points_005, i)

            else:
                print("point not only in box,continue")
                continue
        #end = time.time()
        #print("adapative time:", end - start)
        self.output_image.save("../output/{}_add.tif".format(self.Image_name))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('#current_cnt:{}'.format(len(self.sizedpoints)))
            log_file.write('#add_point:{}'.format(add_points))
        with open(self.run_log_name, "a") as log_file:
            log_file.write('#sized_interact_box_area:{}'.format(crop_area_base))
        with open(self.run_log_name_box, "a") as log_file:
            log_file.write('#sized_box_add:{}'.format(crop_area_base))
            log_file.write('#current_cnt_add:{}'.format(len(self.sizedpoints)))
        # with open(self.run_log_name_add, "a") as log_file:
        #     log_file.write('#current_cnt:{}'.format(len(self.sizedpoints)))


    def interactive_reset(self):
        # if not self.INIT_COUNT_FLAG:
        #     messagebox.showinfo("Initial Counting First", "Please do initial counting first.")
        #     return
        print("interactive_reset")
        self.seletedpoints = []
        self.output_image = self.output_image_back_up.copy()
        self.Visual_image_finding = self.output_image_back_up.copy()
        photo = ImageTk.PhotoImage(self.output_image)
        self.Visual_Label.configure(image=photo)
        self.Visual_Label.image = photo


if __name__ == "__main__":
    win = AICCInterface(None)
    win.mainloop()
