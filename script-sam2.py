import argparse
import json
import torch
import os
import numpy as np
import cv2
import imageio
import sys
import pickle
os.chdir("/workspace/segment-anything-2-real-time-main")
sys.path.append('/workspace/segment-anything-2-real-time-main')
from sam2.build_sam import build_sam2_camera_predictor
from tqdm import tqdm  # 导入 tqdm
checkpoint = "/workspace/segment-anything-2-real-time-main/checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"
predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

#output_video = '/workspace/小莫：看我变变变！莫雷高德VS费利克斯_338.mp4'
with open('/workspace/output_video.pkl', 'rb') as file:
    output_video = pickle.load(file)
print(output_video)
#all_ok_bboxes = [[[444.9430236816406, 12.420000076293945], [1490.510986328125, 1078.3800048828125]]]
# 从文件中加载变量
with open('/workspace/all_ok_bboxes.pkl', 'rb') as file:
    all_ok_bboxes = pickle.load(file)
print(all_ok_bboxes)
out_dir = "/workspace/output_imges"
os.makedirs(out_dir, exist_ok=True)
import shutil
shutil.rmtree('/workspace/output_imges', ignore_errors=True)
os.makedirs('/workspace/output_imges', exist_ok=True)



cap = cv2.VideoCapture(output_video)

if_init = False
n = 1

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)
with tqdm(total=total_frames, desc="Processing frames") as pbar:  # 使用 tqdm 创建进度条
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        width, height = frame.shape[:2][::-1]
        if not if_init:
            predictor.load_first_frame(frame)
            if_init = True
            ann_frame_idx = 0  # the frame index we interact with

            # 遍历所有组的坐标
            for i, bbox_coords in enumerate(all_ok_bboxes):
                ann_obj_id = i + 1  # 对象 ID 从 1 开始
                bbox = np.array(bbox_coords, dtype=np.float32)

                _, out_obj_ids, out_mask_logits = predictor.add_new_prompt(
                    frame_idx=ann_frame_idx, obj_id=ann_obj_id, bbox=bbox
                )

            print(i, bbox_coords,ann_obj_id)
        else:
            out_obj_ids, out_mask_logits = predictor.track(frame)

            all_mask = np.zeros((height, width, 1), dtype=np.uint8)
            for i in range(0, len(out_obj_ids)):
                out_mask = (out_mask_logits[i] > 0.0).permute(1, 2, 0).cpu().numpy().astype(
                    np.uint8
                ) * 255

                all_mask = cv2.bitwise_or(all_mask, out_mask)

            #all_mask = cv2.cvtColor(all_mask, cv2.COLOR_GRAY2RGB)
            #frame = cv2.addWeighted(frame, 1, all_mask, 0.5, 0)
    

            # 假设 all_mask 和 frame 已经定义
            #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
            # 定义腐蚀操作的结构元素
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            # 对 all_mask 进行腐蚀操作
            eroded_mask = cv2.erode(all_mask, kernel, iterations=1)
            all_mask = eroded_mask
            # 对 all_mask 进行膨胀操作
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            dilated_mask = cv2.dilate(all_mask, kernel, iterations=1)
            all_mask = dilated_mask
            #all_mask = cv2.cvtColor(dilated_mask, cv2.COLOR_GRAY2BGR)  # 将 all_mask 转换为三通道图像
            #all_mask = dilated_mask.astype(np.uint8) * 255 
            # 保存掩膜
            #cv2.imwrite("/workspace/output_imges_mask/" + str(n) + ".jpg", all_mask )
            #cv2.imwrite('/content/tem.jpg', all_mask * 255)
            #image = cv2.imread('/content/tem.jpg')
            masked_image = cv2.bitwise_and(frame, frame, mask=all_mask)
            '''模糊
            # 将掩膜应用于原始图片 
            #blurred_image = cv2.GaussianBlur(frame, (21, 21), 500)  # 使用较大的核大小进行模糊
            blurred_image =cv2.medianBlur(frame, 201)
            # 将提取的部分区域叠加到模糊后的图片上
            blurred_image = cv2.bitwise_and(blurred_image, blurred_image, mask=~all_mask)
            # 将提取的部分区域叠加到模糊后的图片上
            frame = np.where(all_mask[:, :, None] > 0, masked_image, blurred_image)
            
                # 创建一个全黑的背景图像
            black_background = np.zeros_like(frame)

            # 将 frame 的透明度设置为 50%
            transparent_frame = cv2.addWeighted(frame, 0.15, black_background, 0.85, 0)

            # 将提取的部分区域叠加到透明后的图片上
            frame = np.where(all_mask[:, :, None] > 0, masked_image, transparent_frame)
            # result 即为只保留 all_mask 遮罩内容的图像
            '''
            # 创建一个全黑的背景图像
            black_background = np.zeros_like(frame)

            # 创建一个全白的背景图像
            white_background = np.ones_like(frame) * 255

            # 将 frame 的透明度设置为 50%
            transparent_frame = cv2.addWeighted(frame, 0.1, black_background, 0.9, 0)

            # 检查每个像素是否为白色，如果是白色，则保持其不透明
            #white_mask = (frame >= 100).all(axis=-1)
            white_mask = (frame >= 130).all(axis=-1)
            frame = np.where(white_mask[:, :, None], frame, transparent_frame)

            # 将提取的部分区域叠加到透明后的图片上
            frame = np.where(all_mask[:, :, None] > 0, masked_image, frame)
            # result 即为只保留 all_mask 遮罩内容的图像
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("/workspace/output_imges/" + str(n) + ".jpg", frame)
        n = n + 1
        pbar.update(1)  # 更新进度条

cap.release()


print('ok')