import cv2
import numpy as np
import os
import dataset
import time
import json
import logging

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

        self.count = min(len(self.image_list), self.count)
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

