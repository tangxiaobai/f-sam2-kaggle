from transformers import AutoProcessor, AutoModelForCausalLM
import copy
import io
from PIL import Image, ImageDraw, ImageFont 
import random
import numpy as np
import pickle
import supervision as sv
# 从文件中加载变量

models = {

    'microsoft/Florence-2-large': AutoModelForCausalLM.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True).to("cuda").eval(),

}

processors = {
 
    'microsoft/Florence-2-large': AutoProcessor.from_pretrained('microsoft/Florence-2-large', trust_remote_code=True)

}


def run_example(task_prompt, image, text_input=None, model_id='microsoft/Florence-2-large'):
    model = models[model_id]
    processor = processors[model_id]
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    return parsed_answer



def convert_to_od_format(data):
    bboxes = data.get('bboxes', [])
    labels = data.get('bboxes_labels', [])
    od_results = {
        'bboxes': bboxes,
        'labels': labels
    }
    return od_results



with open('/workspace/texts.pkl', 'rb') as file:
    texts = pickle.load(file)
print(texts)
with open('/workspace/output_video.pkl', 'rb') as file:
    output_video = pickle.load(file)
print(output_video)
frame_generator = sv.get_video_frames_generator(output_video)
frame = next(frame_generator)
#frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
frame = Image.fromarray(frame)
width, height = frame.size
print(width, height)

detections_list = []
all_ok_bboxes = []
for text in texts:
    print(text)
    task_prompt = '<OPEN_VOCABULARY_DETECTION>'
    text_input = text
    image  = frame
    results = run_example(task_prompt, image, text_input, model_id='microsoft/Florence-2-large')
    bbox_results = convert_to_od_format(results['<OPEN_VOCABULARY_DETECTION>'])
    print(bbox_results)
    half_area = width * height * 0.5

    # 存储所有 the table 的边界框和面积
    table_bboxes = []
    table_areas = []
    given_area =1000
    # 统计 men 的数量
    men_count = 0
    men_label_flag = False
    accflag = False
    for bbox, label in zip(results['<OPEN_VOCABULARY_DETECTION>']['bboxes'], results['<OPEN_VOCABULARY_DETECTION>']['bboxes_labels']):
        if label == 'men':
            men_label_flag = True
            all_ok_bboxes.append([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
            men_count += 1
            print('men add 1',men_count)
        if label == 'ping pong ball':
            # 计算当前 ping pong ball 的面积
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            # 检查面积是否不超过给定边界框的面积
            if area <= given_area:
                all_ok_bboxes.append([[bbox[0], bbox[1]], [bbox[2], bbox[3]]])
        elif label == 'the table':
            # 计算当前 the table 的面积
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            table_bboxes.append([[bbox[0] - 100, bbox[1]], [bbox[2] + 100, bbox[3]]])
            table_areas.append(area)


    # 找到面积最大的 the table
    if table_areas:
        max_area_index = table_areas.index(max(table_areas))
        max_area_bbox = table_bboxes[max_area_index]
        
        # 检查面积是否超过50%
        if max(table_areas) < half_area:
            all_ok_bboxes.append(max_area_bbox)





print(all_ok_bboxes)
# 保存变量到文件
with open('/workspace/all_ok_bboxes.pkl', 'wb') as file:
    pickle.dump(all_ok_bboxes, file)

