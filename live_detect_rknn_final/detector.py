import cv2
import numpy as np
from collections import deque
from rknnlite.api import RKNNLite

import config  # to read CONF_THRESHOLD_GLOBAL


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    interp = cv2.INTER_AREA if r < 1.0 else cv2.INTER_LINEAR
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=interp)
    dw = new_shape[1] - nw
    dh = new_shape[0] - nh
    dw /= 2
    dh /= 2
    img = cv2.copyMakeBorder(
        img,
        int(round(dh - 0.1)), int(round(dh + 0.1)),
        int(round(dw - 0.1)), int(round(dw + 0.1)),
        cv2.BORDER_CONSTANT, value=color
    )
    return img, r, dw, dh


class RKNNAccidentDetector:
    def __init__(self, model_path, conf_threshold=0.5, input_size=640, nms_iou=0.45):
        self.conf_threshold = conf_threshold
        self.nms_iou = nms_iou
        self.prediction_buffer = deque(maxlen=3)
        self.accident_cooldown = 0
        self.cooldown_time = 3
        self.high_confidence = 0.8
        self.input_size = input_size

        self.rknn = RKNNLite()
        print('--> Loading RKNN model')
        assert self.rknn.load_rknn(model_path) == 0, "Failed to load RKNN"
        print('--> Init runtime')
        assert self.rknn.init_runtime() == 0, "Failed to init RKNN runtime"
        print('RKNN model loaded successfully')

        self.ratio = None
        self.dw = None
        self.dh = None
        self.img_shape = None
        self._dbg_once = False

    def preprocess_frame(self, frame):
        self.img_shape = frame.shape[:2]
        img, self.ratio, self.dw, self.dh = letterbox(frame, (self.input_size, self.input_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        img = np.expand_dims(img, 0)
        return img

    @staticmethod
    def nms(boxes, scores, iou_thresh=0.45):
        if len(boxes) == 0:
            return []
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_thresh)[0]
            order = order[inds + 1]
        return keep

    def postprocess(self, outputs, orig_shape):
        preds = np.transpose(outputs[0], (0, 2, 1))[0].astype(np.float32)
        if preds.size == 0:
            return []
        boxes = preds[:, :4].copy()
        raw_scores = preds[:, 4]

        if raw_scores.max() > 1.5 or raw_scores.min() < 0:
            scores = 1 / (1 + np.exp(-raw_scores))
        else:
            scores = raw_scores

        cx, cy, w_box, h_box = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = cx - w_box / 2
        y1 = cy - h_box / 2
        x2 = cx + w_box / 2
        y2 = cy + h_box / 2
        boxes = np.stack([x1, y1, x2, y2], axis=1)

        h, w = orig_shape[:2]
        boxes[:, [0, 2]] -= self.dw
        boxes[:, [1, 3]] -= self.dh
        boxes[:, [0, 2]] /= self.ratio
        boxes[:, [1, 3]] /= self.ratio

        mask = scores > self.conf_threshold
        if not np.any(mask):
            return []
        boxes = boxes[mask]
        scores = scores[mask]

        boxes = np.clip(boxes, 0, [w, h, w, h]).astype(int)
        keep = self.nms(boxes, scores, iou_thresh=self.nms_iou)
        boxes = boxes[keep]
        scores = scores[keep]

        return [(int(a), int(b), int(c), int(d), float(s))
                for (a, b, c, d), s in zip(boxes, scores)]

    def detect(self, frame):
        # sync internal threshold with global web-controlled threshold
        self.conf_threshold = float(config.CONF_THRESHOLD_GLOBAL)

        processed_frame = self.preprocess_frame(frame)
        outputs = self.rknn.inference(inputs=[processed_frame])

        if not self._dbg_once:
            self._dbg_once = True
            arr = outputs[0]
            try:
                preds = np.transpose(arr, (0, 2, 1))[0].astype(np.float32)
                s = preds[:, 4]
                print('RKNN out shape:', arr.shape,
                      'score stats [min max mean]:',
                      float(s.min()), float(s.max()), float(s.mean()))
            except Exception as e:
                print('Debug parse failed:', e)

        detected_boxes = self.postprocess(outputs, frame.shape)
        has_accident = len(detected_boxes) > 0
        max_conf = max([box[4] for box in detected_boxes], default=0.0)

        # temporal smoothing
        self.prediction_buffer.append(1 if has_accident else 0)
        if max_conf > self.high_confidence:
            self.accident_cooldown = self.cooldown_time
            hud_is_acc = True
        elif sum(self.prediction_buffer) >= 2:
            self.accident_cooldown = self.cooldown_time
            hud_is_acc = True
        elif self.accident_cooldown > 0:
            self.accident_cooldown -= 1
            hud_is_acc = True
        else:
            hud_is_acc = False

        return has_accident, max_conf, detected_boxes, hud_is_acc

    def __del__(self):
        try:
            self.rknn.release()
        except Exception:
            pass
