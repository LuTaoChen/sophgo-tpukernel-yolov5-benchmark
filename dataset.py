import logging
import sys
import time
from tpu_perf.infer import SGTypeTuple
import numpy as np

TYPEMAP = {t[1]: t[0] for t in SGTypeTuple}

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dataset")

class Item():
    def __init__(self, label, img, idx):
        self.label = label
        self.img = img
        self.idx = idx
        self.start = time.time()


def usleep(sec):
    if sys.platform == 'win32':
        # on windows time.sleep() doesn't work to well
        import ctypes
        kernel32 = ctypes.windll.kernel32
        timer = kernel32.CreateWaitableTimerA(ctypes.c_void_p(), True, ctypes.c_void_p())
        delay = ctypes.c_longlong(int(-1 * (10 * 1000000 * sec)))
        kernel32.SetWaitableTimer(timer, ctypes.byref(delay), 0, ctypes.c_void_p(), ctypes.c_void_p(), False)
        kernel32.WaitForSingleObject(timer, 0xffffffff)
    else:
        time.sleep(sec)


class Dataset():
    def __init__(self):
        self.arrival = None
        self.image_list = []
        self.label_list = []
        self.input_list_inmemory = {}
        self.last_loaded = -1
        self.model_info = {}

    def preprocess(self, use_cache=True):
        raise NotImplementedError("Dataset:preprocess")

    def get_item_count(self):
        return len(self.image_list)

    def get_list(self):
        raise NotImplementedError("Dataset:get_list")

    def load_query_samples(self, sample_list):
        self.input_list_inmemory = {}
        scale = None
        input_dtype = None
        if self.model_info:
            info = iter(self.model_info.values())
            input_dict = next(info)
            input_dtype = TYPEMAP[input_dict['dtype']]
            scale = input_dict['scale']
        for sample in sample_list:
            self.input_list_inmemory[sample], _ = self.get_item(sample, scale=scale, input_dtype=input_dtype)
        self.last_loaded = time.time()
        print('load query samples done, with {} image inmemory'.format(len(self.input_list_inmemory)))
        time.sleep(0.1)

    def unload_query_samples(self, sample_list=None):
        if sample_list:
            for sample in sample_list:
                if sample in self.input_list_inmemory :
                    del self.input_list_inmemory[sample]
        else:
            self.input_list_inmemory = {}

    def get_samples(self, id_list):
        data = np.array([self.input_list_inmemory[id] for id in id_list])
        return data, self.label_list[id_list]

    def get_item_loc(self, id):
        raise NotImplementedError("Dataset:get_item_loc")
