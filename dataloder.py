import cv2
import numpy as np
import collections
import os
import pickle
import dataset
import time
import json
import logging
from pycocotools.cocoeval import COCOeval
import pycocotools.coco as pycoco

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("coco")


class coco_loader(dataset.Dataset):
    def __init__(self,
                 count,
                 data_path,
                 cache_dir=None,
                 name=None,
                 use_cache=0,
                 image_size=None,
                 image_format="NHWC",
                 pre_process=None,
                 use_label_map=False):
        super().__init__()
        print("Constructing Data loader...")
        self.count = count
        self.input_features = []
        self.image_size = image_size
        self.image_list = []
        self.label_list = []
        self.image_ids = []
        self.image_sizes = []
        self.data_path = data_path
        self.use_cache = use_cache
        self.pre_process = pre_process
        self.use_label_map = use_label_map

        if not cache_dir:
            cache_dir = os.getcwd()
        self.cache_dir = os.path.join(cache_dir, "preprocessed", name, image_format)
        # input images are in HWC
        self.need_transpose = True if image_format == "NCHW" else False
        not_found = 0
        empty_80catageories = 0
        image_list = os.path.join(data_path, "annotations/instances_val2017.json")
        self.annotation_file = image_list
        if self.use_label_map:
            # for pytorch
            label_map = {}
            with open(self.annotation_file) as fin:
                annotations = json.load(fin)
            for cnt, cat in enumerate(annotations["categories"]):
                label_map[cat["id"]] = cnt + 1

        os.makedirs(self.cache_dir, exist_ok=True)
        start = time.time()
        images = {}
        with open(image_list, "r") as f:
            coco = json.load(f)
        for i in coco["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}
        for a in coco["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            catagory_ids = label_map[a.get("category_id")] if self.use_label_map else a.get("category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))

        for image_id, img in images.items():
            image_name = os.path.join("images/val2017", img["file_name"])
            src = os.path.join(data_path, image_name)
            if not os.path.exists(src):
                # if the image does not exists ignore it
                not_found += 1
                continue
            if len(img["category"]) == 0 and self.use_label_map:
                # if an image doesn't have any of the 81 categories in it
                empty_80catageories += 1  # should be 48 images - thus the validation sert has 4952 images
                continue

            os.makedirs(os.path.dirname(os.path.join(self.cache_dir, image_name)), exist_ok=True)
            dst = os.path.join(self.cache_dir, image_name)
            if not os.path.exists(dst + ".npy"):
                # cache a preprocessed version of the image
                img_org = cv2.imread(src)
                processed = self.pre_process(img_org, need_transpose=self.need_transpose, dims=self.image_size)
                np.save(dst, processed)

            self.image_ids.append(image_id)
            self.image_list.append(image_name)
            self.image_sizes.append((img["height"], img["width"]))
            self.label_list.append((img["category"], img["bbox"]))

            # limit the dataset if requested
            if self.count and len(self.image_list) >= self.count:
                break

        time_taken = time.time() - start
        if not self.image_list:
            log.error("no images in image list found")
            raise ValueError("no images in image list found")
        if not_found > 0:
            log.info("reduced image list, %d images not found", not_found)
        if empty_80catageories > 0:
            log.info("reduced image list, %d images without any of the 80 categories", empty_80catageories)

        log.info("loaded {} images, cache={}, took={:.1f}sec".format(
            len(self.image_list), use_cache, time_taken))

        self.label_list = np.array(self.label_list, dtype=object)

    def __len__(self):
        return self.count

    def get_item(self, nr, scale=None, input_dtype=None):
        """Get image by number in the list."""
        dst = os.path.join(self.cache_dir, self.image_list[nr])
        img = np.load(dst + ".npy")
        if scale and input_dtype:
            img = (img * scale).astype(input_dtype)
        return img, self.label_list[nr]

    def __getitem__(self, idx):
        return self.input_list_inmemory[idx]

    def __iter__(self):
        pass

    def __next__(self):
        pass

def resize_with_aspectratio_padding(img, out_height, out_width, inter_pol=cv2.INTER_LINEAR, color=(114, 114, 114)):
    height, width, _ = img.shape
    if width > height:
        w = out_width
        h = int(out_height * height / width)
    else:
        h = out_height
        w = int(out_width * width / height)
    img = cv2.resize(img, (w, h), interpolation=inter_pol)

    dw, dh = out_width - w, out_height - h
    dw /= 2
    dh /= 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img

def pre_process_coco_yolov5(img, dims=None, need_transpose=False):
    new_h, new_w, _ = dims
    img = resize_with_aspectratio_padding(img, new_h, new_w)
    img = img / 255.0

    if need_transpose:
        img = img.transpose([2, 0, 1])  # HWC to CHW,
    img = np.ascontiguousarray(img[::-1])  # contiguous BGR to RGB
    return img


def get_dataloader(count,
                   input_file,
                   cache_path,
                   name,
                   image_format,
                   pre_process,
                   use_cache, **kwargs):

    return coco_loader(count=count,
                       data_path=input_file,
                       cache_dir=cache_path,
                       name=name,
                       use_cache=use_cache,
                       image_format=image_format,
                       pre_process=pre_process,
                       **kwargs)


class PostProcessCocoYolo:
    """
    Postprocessing required by yolov5
    """
    def __init__(self, use_inv_map, score_threshold=0.5):
        super().__init__()
        self.use_inv_map = use_inv_map
        self.score_threshold = score_threshold
        self.results = []
        self.good = 0
        self.total = 0
        self.content_ids = []

    def add_results(self, results):
        self.results.extend(results)

    def __call__(self, results, ids, expected=None, result_dict=None, label_offset=0):
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

    def start(self):
        self.results = []
        self.good = 0
        self.total = 0

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
        for batch in range(0, len(self.results)):
            image_indices.append(self.content_ids[batch])
            if save_result:
                path = ds.data_path + '/' + ds.image_list[self.content_ids[batch]]
                img = cv2.imread(path)
            for idx in range(0, len(self.results[batch])):
                detection = self.results[batch][idx]
                # this is the index of the coco image
                image_idx = int(detection[0])
                if image_idx != self.content_ids[batch]:
                    # working with the coco index/id is error prone - extra check to make sure it is consistent
                    log.error("image_idx missmatch, lg={} / result={}".format(image_idx, self.content_ids[batch]))
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
                ret = cv2.imwrite(output_dir + '/' + ds.image_list[self.content_ids[batch]][-16:], img)

        # map indices to coco image id's
        image_ids = [ds.image_ids[i] for i in image_indices]
        self.results = []
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
