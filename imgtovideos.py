# @title #处理完了生成视频
import cv2
import os
import re
import pickle
from moviepy.editor import VideoFileClip, ImageSequenceClip

def create_video_with_audio(image_folder, input_video_path):
    # 获取图像文件列表
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 自然排序图像文件
    def natural_sort_key(s, _nsre=re.compile('([0-9]+)')):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(_nsre, s)]

    image_files.sort(key=natural_sort_key)

    # 跳过第一张图片
    if image_files and len(image_files) != 1:
        image_files = image_files[1:]

    # 读取第一张图像以获取尺寸
    if image_files:
        first_image = cv2.imread(os.path.join(image_folder, image_files[0]))
        height, width, layers = first_image.shape
    else:
        raise ValueError("No valid images found in the folder after skipping the first one.")

    # 获取输入视频的帧率
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # 创建图像序列视频
    image_paths = [os.path.join(image_folder, img) for img in image_files]
    clip = ImageSequenceClip(image_paths, fps=fps)

    # 从输入视频中提取音频
    audio_clip = VideoFileClip(input_video_path).audio

    # 将音频添加到视频中
    final_clip = clip.set_audio(audio_clip)

    # 生成与输入视频同名的输出文件
    output_video_path = os.path.join('/workspace/okvideos', os.path.basename(input_video_path))
    #output_video_path = os.path.join('/tmp/', os.path.basename(input_video_path))
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # 导出最终视频
    final_clip.write_videofile(output_video_path, codec='libx264')

    print(f"Video created successfully: {output_video_path}")
    return output_video_path

# 示例调用
image_folder = '/workspace/output_imges'
with open('/workspace/output_video.pkl', 'rb') as file:
    output_video = pickle.load(file)
print(output_video)
input_video_path = output_video

create_video_with_audio(image_folder, input_video_path)
''''''