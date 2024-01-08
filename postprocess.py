import functools
import cv2
import numpy as np
import time
import json
from pycocotools.cocoeval import COCOeval
import pycocotools.coco as pycoco
import logging

log = logging.getLogger("postprocess")


def cpu_postprocess_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        self, results, ids, _ = args[0], args[1], args[2], args[3]
        if len(results) == 1 and len(results[0].shape) == 3:
            pred = non_max_suppression(results[0], conf_thres=self.conf_thres, iou_thres=self.iou_thres, classes=None)
            det_boxes = []
            for i, p in enumerate(pred):
                if p is None:
                    log.info("img id {} has no dectection".format(ids[i]))
                    continue
                else:
                    sample_ids = np.ones((p.shape[0], 1)) * ids[i]
                    self.content_ids.extend([ids[i]] * p.shape[0])
                    det_box = np.concatenate((sample_ids, p), axis=-1)
                    det_boxes.extend(list(det_box))
            return det_boxes
        else:
            return func(*args, **kwargs)
    return wrapper


ANCHORS = np.array([
    [10,  13, 16,  30,  33,  23 ],
    [30,  61, 62,  45,  59,  119],
    [116, 90, 156, 198, 373, 326]
])
ANCHOR_GRID = ANCHORS.reshape(3, -1, 2).reshape(3, 1, -1, 1, 1, 2)
STRIDES = [8, 16, 32]


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


def make_grid(nx, ny):
    z = np.stack(np.meshgrid(np.arange(nx), np.arange(ny)), 2)
    return z.reshape(1, 1, ny, nx, 2).astype(np.float32)


def predict_preprocess(x):
    for i in range(len(x)):
        bs, na, ny, nx, no = x[i].shape
        grid = make_grid(nx, ny)
        x[i] = sigmoid(x[i])
        x[i][..., 0:2] = (x[i][..., 0:2] * 2. - 0.5 + grid) * STRIDES[i]
        x[i][..., 2:4] = (x[i][..., 2:4] * 2) ** 2 * ANCHOR_GRID[i]
        x[i] = x[i].reshape(bs, -1, no)
    return np.concatenate(x, 1)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def _nms(dets, scores, prob_threshold):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    score_index = np.argsort(scores)[::-1]
    keep = []
    while score_index.size > 0:
        max_index = score_index[0]
        keep.append(max_index)
        xx1 = np.maximum(x1[max_index], x1[score_index[1:]])
        yy1 = np.maximum(y1[max_index], y1[score_index[1:]])
        xx2 = np.minimum(x2[max_index], x2[score_index[1:]])
        yy2 = np.minimum(y2[max_index], y2[score_index[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        union = width * height
        iou = union / (areas[max_index] + areas[score_index[1:]] - union)
        ids = np.where(iou < prob_threshold)[0]
        score_index = score_index[ids+1]
    return keep


def non_max_suppression(prediction, conf_thres=0.001, iou_thres=0.3, classes=None):
    if len(prediction) == 3:
        prediction = [prediction['1'], prediction['2'], prediction['3']]
        prediction = predict_preprocess(prediction)
    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = np.stack((x[:, 5:] > conf_thres).nonzero())
            x = np.concatenate((box[i], x[i, j + 5, None], j[:, None].astype(np.float32)), 1)
        else:  # best class only
            conf = x[:, 5:].max(1, keepdims=True)
            j = x[:, 5:].argmax(1).reshape(-1, 1)
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == np.array(classes)).any(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = _nms(boxes, scores, iou_thres)
        if len(i) > max_det:  # limit detections
            i = i[:max_det]

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded
        output[xi] = np.concatenate((output[xi][:, 0:2],
                                     output[xi][:, 2:4] - output[xi][:, 0:2],
                                     output[xi][:, 4:5],
                                     output[xi][:, 5:6] + 1),
                                    axis=-1)
    return output


class PostProcessCocoYolo:
    """
    Postprocessing required by yolov5
    """
    def __init__(self, use_inv_map, conf_thres=0.001, iou_thres=0.45):
        super().__init__()
        self.use_inv_map = use_inv_map
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

    def add_results(self, results):
        self.results.extend(list(results))

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

    # return [[image_id, x, y, w, h, prob, cls]]
    @cpu_postprocess_call
    def __call__(self, results, ids, expected=None, result_dict=None):
        processed_results = []
        if len(results) == 2:
            batch_result = results[0]
            dt_num = results[1][0]
        elif len(results) == 1:
            dt_num = len(results[0][0, 0])
            batch_result = results[0]

        det_boxes = batch_result[0, 0]
        for i in range(len(det_boxes)):
            det_box = det_boxes[None, i]
            if dt_num == 0 or (len(det_box) == 1 and det_box[0, 0] == -1):
                raise ValueError
            xywh, prob, cls = det_box[:, 3:7], det_box[:, 2:3], det_box[:, 1:2] + 1
            xywh = np.concatenate((xywh[:, 0:2] - xywh[:, 2:] / 2, xywh[:, 2:]), axis=1)
            bidx = int(det_box[0, 0])
            det_box = np.concatenate((np.ones_like(cls) * ids[bidx], xywh, prob, cls), axis=1)
            processed_results.append(det_box)
            expected_classes = expected[bidx][0]
            self.content_ids.append(ids[bidx])
            detection_class = int(det_box[0, 6])
            if detection_class in expected_classes:
                self.good += 1
            self.total += 1
        return processed_results

    def finalize(self, result_dict, ds=None, output_dir=None, save_result=False):
        result_dict["good"] += self.good
        result_dict["total"] += self.total

        if self.use_inv_map:
            # for pytorch
            label_map = {}
            with open(ds.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1
            inv_map = {v: k for k, v in label_map.items()}

        detections = []
        image_indices = []
        for boxid in range(0, len(self.results)):
            image_indices.append(self.content_ids[boxid])
            if save_result:
                path = ds.data_path + '/' + ds.image_list[self.content_ids[boxid]]
                img = cv2.imread(path)

            detection = self.results[boxid].squeeze()
            # this is the index of the coco image
            image_idx = int(detection[0])
            if image_idx != self.content_ids[boxid]:
                # working with the coco index/id is error prone - extra check to make sure it is consistent
                log.error("image_idx missmatch, lg={} / result={}".format(image_idx, self.content_ids[boxid]))
            # map the index to the coco image id
            detection[0] = ds.image_ids[image_idx]
            height, width = ds.image_sizes[image_idx]
            # pycoco wants {imageID,x1,y1,w,h,score,class}
            detection[3] *= max(width, height) / max(ds.image_size[0], ds.image_size[1])
            detection[4] *= max(width, height) / max(ds.image_size[0], ds.image_size[1])
            detection[1] *= max(width, height) / max(ds.image_size[0], ds.image_size[1])
            detection[2] *= max(width, height) / max(ds.image_size[0], ds.image_size[1])

            if width < height:
                detection[1] -= (height - width) / 2
            else:
                detection[2] -= (width - height) / 2

            if self.use_inv_map:
                cat_id = inv_map.get(int(detection[6]), -1)
                if cat_id == -1:
                    # FIXME:
                    log.info("finalize can't map category {}".format(int(detection[6])))
                detection[6] = cat_id

            if save_result:
                xmax = detection[1] + detection[3]
                ymax = detection[2] + detection[4]
                box = [detection[1], detection[2], xmax, ymax]
                img = self.draw_rectangle(img, box, str(cat_id))
            detections.append(np.array(detection))
        if save_result:
            ret = cv2.imwrite(output_dir + '/' + ds.image_list[self.content_ids[boxid]][-16:], img)

        # map indices to coco image id's
        image_ids = [ds.image_ids[i] for i in image_indices]
        cocoGt = pycoco.COCO(ds.annotation_file)
        cocoDt = cocoGt.loadRes(np.array(detections))
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        cocoEval.params.imgIds = image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        result_dict["mAP"] = cocoEval.stats[0]

    def draw_rectangle(self, img, box, label='', color=(128, 0, 128), txt_color=(255, 255, 255), path=None):
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        lw = max(round(sum(img.shape) / 2 * 0.003), 2)
        cv2.rectangle(img, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        tf = max(lw - 1, 1)  # font thickness
        w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(img, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img,
                    label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0,
                    lw / 3,
                    txt_color,
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        return img
