"""
Author: PineSeed
GitHub: https://github.com/Pine-Seed
"""
import time
import numpy as np
import cv2
import tensorflow as tf
from utils import preprocess_img, single_class_non_max_suppression, decode_bbox, generate_anchors, draw_result


# 定义输入输出shape
input_shape = (1, 260, 260, 3)  # image_shape
output_shape = [(1, 5972, 4), (1, 5972, 2)]  # loc_shape, cls_shape

# 定义一些参数
model_path = "./face_mask_detection.tflite"
id2class = {0: "Mask", 1: "No Mask"}
anchors_exp = np.expand_dims(generate_anchors(), 0)
result_format = "Avg FPS:{:^4.2f}, OnTime FPS:{:^4.2f}."

interpreter = tf.lite.Interpreter(model_path=model_path)  # 可指定线程数: num_threads=4
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0)
cost_time = 0.1
while True:
    ret, image = cap.read()
    if not ret:
        break
    if image is None:
        print("未抓取到图像，请检查摄像头是否正常工作")
        time.sleep(0.5)
        continue

    # 水平翻转(似乎会影响到判断)
    image = cv2.flip(image, 1)

    t0 = time.time()
    # 预处理图像
    input_data = preprocess_img(image, target_shape=(input_shape[1], input_shape[2]))
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], input_data)
    interpreter.invoke()

    # 检索模型输出
    loc_res = interpreter.get_tensor(interpreter.get_output_details()[0]['index']).reshape(*output_shape[0])
    cls_res = interpreter.get_tensor(interpreter.get_output_details()[1]['index']).reshape(*output_shape[1])[0]

    # 后处理
    y_bboxes = decode_bbox(anchors_exp, loc_res)[0]
    bbox_max_scores = np.max(cls_res, axis=1)
    bbox_max_score_classes = np.argmax(cls_res, axis=1)
    keep_idxes = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=0.6, iou_thresh=0.4)

    # 画出来
    image = draw_result(image, keep_idxes, y_bboxes, bbox_max_scores, bbox_max_score_classes, id2class)
    t1 = time.time()

    # 计算FPS
    cost_time = 0.8 * cost_time + 0.2 * (t1 - t0)
    fps_text = result_format.format(1 / cost_time, 1 / (t1 - t0))
    cv2.putText(image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 使用OpenCV显示图像
    cv2.imshow('Author - PineSeed', image)

    # 按q退出(小写)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
