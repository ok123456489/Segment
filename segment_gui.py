# -*- coding: utf-8 -*-
import os.path
import time

import pydicom
from PyQt5.QtCore import Qt
from PyQt5.QtGui import (
    QBrush,
    QPainter,
    QPen,
    QPixmap,
    QKeySequence,
    QPen,
    QBrush,
    QColor,
    QImage,
)
from PyQt5.QtWidgets import (
    QFileDialog,
    QApplication,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QGraphicsPixmapItem,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QShortcut,
)

import numpy as np
from skimage import transform, io
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image

from segment_anything import sam_model_registry
#设置随机数种子，清楚gpu显存
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)


#模型种类
SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
MEDSAM_IMG_INPUT_SIZE = 1024


#使用gpu进行数据处理
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#执行模型推理
@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    #处理输入框
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    #sparse_embeddings：编码用户提供的边界框信息，告诉模型需要分割的目标区域。
    #dense_embeddings：编码图像的位置编码或其他全局信息，帮助模型更好地理解上下文。
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    #通过 mask_decoder 模块生成低分辨率的掩码 logits
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )


    #将 logits 转换为概率值，范围在 [0, 1] 之间。
    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    #将低分辨率掩码插值到原始图像的分辨率。
    low_res_pred = F.interpolate(
        low_res_pred,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)

    #去除多余的维度，得到形状为 (height, width) 的数组。squeeze：将维度为1的部分去除
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)


    #将 PyTorch 张量转换为 NumPy 数组，并进行二值化处理。
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg



print("Loading MedSAM model, a sec.")
tic = time.perf_counter()


# 构建模型
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

print(f"Done, took {time.perf_counter() - tic}")

#将np数组转为QPixmap 对象，用于gui输出
def np2pixmap(np_img):
    #获取 NumPy 数组的形状信息。
    height, width, channel = np_img.shape

    #计算图像每行的字节数，RGB图像每个像素由三个字节组成
    bytesPerLine = 3 * width

    #创建 QImage 对象
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)

#掩码颜色
color = (255, 255, 255)



class Window(QWidget):
    def __init__(self):
        super().__init__()
        # 配置
        self.half_point_size = 5  # 边界框起始点和结束点的半径

        # 状态变量
        self.image_path = None # 图像路径

        self.img = None # 原图像
        self.seg_pic = None  # 截取后图片

        self.is_mouse_down = False # 鼠标是否按下
        self.rect = None # 绘制的矩形
        self.point_size = self.half_point_size * 2 # 点的大小
        self.start_point = None # 起始点
        self.end_point = None # 结束点
        self.start_pos = (None, None) # 起始位置
        self.embedding = None # 图像嵌入（编码后的特征图）
        self.mask = None #掩码
        self.initUI()

    def initUI(self):
        # 创建 QGraphicsView 并启用抗锯齿
        self.pic_view = QGraphicsView()
        self.pic_view.setRenderHint(QPainter.Antialiasing)
        self.seg_view = QGraphicsView()
        self.seg_view.setRenderHint(QPainter.Antialiasing)
        #加载图片
        self.load_image()

        # 创建水平布局并添加图片框
        hbox1 = QHBoxLayout()
        hbox1.addWidget(self.pic_view)
        hbox1.addWidget(self.seg_view)

        #读取/保存按钮
        load_button = QPushButton("读取图片")
        save_button = QPushButton("保存截取图片")

        #创建水平布局，并且添加按键
        hbox2 = QHBoxLayout()
        hbox2.addWidget(load_button)
        hbox2.addWidget(save_button)

        #创建垂直布局
        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)  # 添加上半部分（两个 QGraphicsView）
        vbox.addLayout(hbox2)  # 添加下半部分（两个按钮）

        #设置窗口主布局
        self.setLayout(vbox)

        self.setWindowTitle("segment_gui")
        self.resize(600, 400)
        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_seg_pic)

    #读取DICOM中的图片
    def dcm_to_png(self,file_path):
        # 读取 DICOM 文件
        dicom_data = pydicom.dcmread(file_path)

        # 提取像素数据数组
        pixel_array = dicom_data.pixel_array

        # 如果 DICOM 文件包含元数据，可能需要调整像素值
        if hasattr(dicom_data, "RescaleSlope") and hasattr(dicom_data, "RescaleIntercept"):
            pixel_array = pixel_array * dicom_data.RescaleSlope + dicom_data.RescaleIntercept

        # 将像素值归一化到 0-255 范围，防止出现全黑或者全白
        pixel_array = ((pixel_array - pixel_array.min()) / (pixel_array.max() - pixel_array.min()) * 255).astype(
            np.uint8)

        # 创建 PIL 图像对象
        image = Image.fromarray(pixel_array)
        return image

    #加载图片
    def load_image(self):
        #加载图片
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp *.dcm)"
        )

        #没有选择图片
        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        #选择dcm格式数据，先进行图片读取
        if file_path.lower().endswith('.dcm') :
            img_np = self.dcm_to_png(file_path)
            #转换为np数组类型
            img_np = np.array(img_np)

        #读取图片
        else:
            img_np = io.imread(file_path)

        # 将单通道图像转换为三通道
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        #三通道图片
        self.img_3c = img_3c
        #图片路径
        self.image_path = file_path

        #获取特征图
        self.get_embeddings()

        # 转化np数组为可输出的图片
        pic_pixmap = np2pixmap(self.img_3c)

        #获取图片shaper（H,W,C）
        H, W, _ = self.img_3c.shape

        #图片框
        self.pic_scene = QGraphicsScene(0, 0, W, H)
        #截取后的图片框
        self.seg_scene = QGraphicsScene(0, 0, W, H)

        #用于存储鼠标拖动时绘制的结束点
        self.end_point = None

        #用于存储鼠标拖动时绘制的矩形。
        self.rect = None


        #展示原图
        self.img = self.pic_scene.addPixmap(pic_pixmap)
        self.img.setPos(0, 0)

        self.mask = np.zeros_like(self.img_3c)

        # 截取结果初始化
        self.seg_pic = np.full((*self.img_3c.shape[:2], 3), 255, dtype="uint8")
        seg_pixmap = np2pixmap(self.seg_pic)
        self.seg_pic = self.seg_scene.addPixmap(seg_pixmap)
        self.seg_pic.setPos(0, 0)


        self.pic_view.setScene(self.pic_scene)
        self.seg_view.setScene(self.seg_scene)


        # 事件
        self.pic_scene.mousePressEvent = self.mouse_press
        self.pic_scene.mouseMoveEvent = self.mouse_move
        self.pic_scene.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        self.is_mouse_down = True
        self.start_pos = ev.scenePos().x(), ev.scenePos().y()
        self.start_point = self.pic_scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

    def mouse_move(self, ev):
        if not self.is_mouse_down:
            return

        x, y = ev.scenePos().x(), ev.scenePos().y()

        if self.end_point is not None:
            self.pic_scene.removeItem(self.end_point)
        self.end_point = self.pic_scene.addEllipse(
            x - self.half_point_size,
            y - self.half_point_size,
            self.point_size,
            self.point_size,
            pen=QPen(QColor("red")),
            brush=QBrush(QColor("red")),
        )

        if self.rect is not None:
            self.pic_scene.removeItem(self.rect)
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)
        self.rect = self.pic_scene.addRect(
            xmin, ymin, xmax - xmin, ymax - ymin, pen=QPen(QColor("red"))
        )

    def mouse_release(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        sx, sy = self.start_pos
        xmin = min(x, sx)
        xmax = max(x, sx)
        ymin = min(y, sy)
        ymax = max(y, sy)

        self.is_mouse_down = False


        #获取shape
        H, W, _ = self.img_3c.shape

        #将提示框转为np类型
        box_np = np.array([[xmin, ymin, xmax, ymax]])
        # print("bounding box:", box_np)

        #缩放边界框至（1024 x 1024）
        box_1024 = box_np / np.array([W, H, W, H]) * 1024

        #获取推理后的掩码
        sam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)


        #sam_mask 中非零的区域（即掩码区域）设置为指定的颜色 color，并将结果存储在 self.mask
        self.mask[sam_mask != 0] = color

        # #将掩码与原图拼接

        # 将 NumPy 数组转换为 PIL 图像
        bg = Image.fromarray(self.img_3c.astype("uint8"), "RGB")
        mask = Image.fromarray(self.mask.astype("uint8"), "RGB")

        # 将背景图像和掩码转换为 NumPy 数组以便处理
        bg_np = np.array(bg)
        mask_np = np.array(mask)

        # 创建一个与背景图像相同大小的白色背景
        result_np = np.zeros_like(bg_np)  # 创建一个全0数组

        # 在结果图像中根据掩膜填充背景图像
        result_np[mask_np > 0] = bg_np[mask_np > 0]  # 使用掩膜进行切割

        # 将结果转换回 PIL 图像
        self.seg_pic = Image.fromarray(result_np.astype("uint8"), "RGB")
        # 将结果转换为 NumPy 数组
        self.seg_pic_np = np.array(self.seg_pic)

        #查看seg_pic是否存在于seg_scene中！！不存在删除会闪退
        if self.seg_pic in self.seg_scene.items():
            self.seg_scene.removeItem(self.seg_pic)

        self.seg_pic = self.seg_scene.addPixmap(np2pixmap(self.seg_pic_np))

    #保存图片
    def save_seg_pic(self):
        directory = os.path.dirname(self.image_path)
        root_folder_name = os.path.basename(directory)
        folder_name = root_folder_name + "_seg_result"
        save_path = fr".\PCR\{root_folder_name}\{folder_name}"
        file_name = os.path.splitext(os.path.basename(self.image_path))[0]  # 获取不带扩展名的文件名
        output_file_name = f"{file_name}_seg_pic.png"  # 生成新的文件名
        if os.path.isdir(save_path):
            out_path = os.path.join(save_path,output_file_name)
        else:
            os.makedirs(save_path)
            out_path = os.path.join(save_path, output_file_name)

        print(type(self.seg_pic))
        if isinstance(self.seg_pic, Image.Image):
            seg_pic_np = np.array(self.seg_pic)
        else:
            seg_pic_np = self.seg_pic_np
        io.imsave(out_path,seg_pic_np)

    @torch.no_grad()
    def get_embeddings(self):
        print("Calculating embedding, gui may be unresponsive.")
        #resize图片
        img_1024 = transform.resize(
            self.img_3c,
            (1024, 1024),
            order=3, # 双三次插值
            preserve_range=True, # 保持输入图像的值范围
            anti_aliasing=True # 是否启用抗锯齿
        ).astype(np.uint8)

        #归一化数值至[0, 1]，shape为(H, W, 3)
        #使用最小最大归一化化
        #np.clip(..., a_min=1e-8, a_max=None)：
        #对差值进行裁剪，确保分母不为 0。
        #如果差值小于 1e-8，则将其设置为 1e-8，避免除以 0 的错误。
        #如果差值大于 1e-8，则保持原值。

        img_1024 = (img_1024 - img_1024.min()) / np.clip(
            img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
        )

        # 转换shape为 (3, H, W)
        img_1024_tensor = (
            torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        )

        # 获取特征图
        with torch.no_grad():
            self.embedding = medsam_model.image_encoder(
                img_1024_tensor
            )  # (1, 256, 64, 64)
        print("Done.")


app = QApplication([])

w = Window()
w.show()

app.exec()
