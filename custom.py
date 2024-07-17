import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transform
import torchvision.transforms.functional as tf

from core.config import cfg, cfg_from_file, cfg_from_list
from models import get_model
from datasets import get_num_classes
from infer_val import convert_dict, mask_overlay

from datasets.dataloader_base import DLBase

# 原代码中的 DataLoader 用于批量处理，项目是连续单张图片处理，所以重写一个
class DLInferSingle(DLBase):
    def __init__(self, cfg):
        super(DLInferSingle, self).__init__()
        self.cfg = cfg

    def denorm(self, image):
        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)
        return image


class Dasac(object):
    def __init__(self, args):
        # reading the config
        cfg_from_file(args['cfg_file'])
        if args['set_cfgs'] is not None:
            cfg_from_list(args['set_cfgs'])
        num_classes = get_num_classes(args)

        # Loading the model
        self.model = get_model(cfg.MODEL, 0, num_classes=num_classes)
        assert os.path.isfile(args['resume']), "Snapshot not found: {}".format(args['resume'])
        state_dict = convert_dict(torch.load(args['resume'])["model"])
        # print(self.model)
        self.check_model_dtype()
        print('Loading model from {}'.format(args['resume']))
        self.model.load_state_dict(state_dict, strict=False)

        for p in self.model.parameters():
            p.requires_grad = False

        # setting the evaluation mode
        self.model.eval()
        self.model = nn.DataParallel(self.model).cuda()
        # self.model = nn.DataParallel(self.model).cpu()

        # self.infer_dataset = get_dataloader(args['dataloader'], cfg, args['infer_list'])
        self.infer_dataset = DLInferSingle(cfg)
        self.palette = self.infer_dataset.get_palette()
        # 归一化参数是数据集的均值和标准差
        self.image_norm = transform.Normalize(self.infer_dataset.MEAN, self.infer_dataset.STD)
    
    def infer(self, image):
        image = tf.to_tensor(image)
        image = self.image_norm(image)
        # 原来传进 DataLoader 的时候升高了一维，需要包裹一个 batch_size = 1
        image = image.view(1,*image.size())

        with torch.no_grad():
            _, logits = self.model(image, teacher=False)
            masks_pred = F.softmax(logits, 1)

        mask_pred = masks_pred[0].cpu().numpy()
        mask_pred = np.argmax(mask_pred, 0).astype(np.uint8)

        return mask_pred

    def get_image_overlay(self, image, mask):
        overlay = mask_overlay(mask, image, self.palette)
        image_overlay = cv2.cvtColor(np.asarray((overlay * 255.).astype(np.uint8)), cv2.COLOR_RGB2BGR)
        return image_overlay
    
    def preprocess_image(self, image):
        image = image.astype(np.float32) / 255.0
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    

    def check_model_dtype(self):
        # 初始化计数器
        int8_count = 0
        fp16_count = 0
        fp32_count = 0

        # 遍历模型的所有参数
        for p in self.model.parameters():
            # 检查数据类型
            # print(f"参数名: {p.name}, 数据类型: {p.dtype}")
            if p.dtype == torch.int8:
                int8_count += 1
            elif p.dtype == torch.float16:
                fp16_count += 1
            elif p.dtype == torch.float32:
                fp32_count += 1


        # 打印结果
        print(f"INT8 参数数量: {int8_count}")
        print(f"FP16 参数数量: {fp16_count}")
        print(f"FP32 参数数量: {fp32_count}")

        # 如果需要，也可以返回这些计数
        return int8_count, fp16_count