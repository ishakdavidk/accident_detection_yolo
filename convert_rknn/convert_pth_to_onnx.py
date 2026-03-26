
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
import cv2

# model = YOLO('../runs/train/accident_detection_yolo8_finetune/weights/epoch56.pt')
model = YOLO('yolov11s_epoch77_frozen2.pt')
# model.export(format='onnx', imgsz=640, opset=13, simplify=False)
model.export(format='onnx', imgsz=640, opset=12, simplify=False, dynamic=False)

img = np.ones((640, 640, 3), dtype=np.uint8)*255  # white image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)/255.0
img = np.transpose(img, (2, 0, 1))[None]

# sess = ort.InferenceSession('../runs/train/accident_detection_yolo8_finetune/weights/epoch56.onnx', providers=['CPUExecutionProvider'])
sess = ort.InferenceSession('yolov11s_epoch77_frozen2.onnx', providers=['CPUExecutionProvider'])
out = sess.run(None, {sess.get_inputs()[0].name: img})[0]
print(out.shape, out[0,4,:10])