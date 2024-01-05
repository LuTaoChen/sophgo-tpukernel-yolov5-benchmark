import threading
import time
import logging

import numpy as np
from queue import Queue
from tqdm import trange

# import by other means
# from python.tpu_perf.infer import SGInfer

from tpu_perf.infer import SGInfer, sgtype, nptype

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("main")


class Item:
    """An item that we queue for processing by the thread pool."""

    def __init__(self, query_id, content_index, data, label=None):
        self.query_id = query_id
        self.content_index = content_index
        self.img = data
        self.label = label
        self.start = time.time()

    def iter(self):
        return zip(self.query_id, self.content_index, self.data, self.label)


class RunnerBase:
    def __init__(self, model, devices, ds, threads, post_proc=None, tpu_kernel=None):
        self.take_accuracy = False
        self.ds = ds
        self.model = model
        if tpu_kernel:
            # tpu_kernel = "tpu_kernel_api_yolov5_detect_out"
            self.runner = SGInfer(model, 1, devices=devices, tpu_kernel=tpu_kernel)
        else:
            self.runner = SGInfer(model)
        self.post_process = post_proc
        self.threads = threads
        self.take_accuracy = False
        self.info = self.runner.get_input_info()
        info = iter(self.info.values())
        self.max_batchsize = next(info)['shape'][0]
        self.out_info = self.runner.get_output_info()
        self.result_timing = []
        self.waiting_queries = {}
        self.respond_cond = threading.Condition()

    def handle_tasks(self, tasks_queue):
        pass

    def start_run(self, result_dict, take_accuracy):
        self.result_dict = result_dict
        self.result_timing = []
        self.take_accuracy = take_accuracy
        self.post_process.start()

    def process_result(self, qitem, task_id, results, valid):
        try:
            if not valid:
                raise RuntimeError("sgInfer exception, error {}".format(valid))
            processed_results = self.post_process(results, qitem.content_index, qitem.label,
                                                  self.result_dict)
            if self.take_accuracy:
                self.post_process.add_results(processed_results)
            took = time.time() - qitem.start
            for _ in qitem.query_id:
                self.result_timing.append(took)
        except Exception as ex:
            log.error("Failed to postprocess task_id: {}".format(task_id))
        finally:
            for idx, query_id in enumerate(qitem.query_id):
                self.waiting_queries.pop(query_id)
            if len(self.waiting_queries) == 0:
                with self.respond_cond:
                    self.respond_cond.notify()

    def run_one_item(self, qitem):
        # run the prediction
        processed_results = []
        task_id = self.runner.put(qitem.img)
        res = self.runner.get()
        if res[0] != task_id:
            log.error('task id dismatch {} vs {}'.format(task_id, res[0]))
        self.process_result(qitem, *res)

    def enqueue(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            self.run_one_item(Item(query_id, idx, data, label))
        else:
            bs = self.max_batchsize
            for i in range(0, len(idx), bs):
                data, label = self.ds.get_samples(idx[i:i+bs])
                self.run_one_item(Item(query_id[i:i+bs], idx[i:i+bs], data, label))

    def finish(self):
        pass

    def wait_for_response(self):
        with self.respond_cond:
            self.respond_cond.wait()


class QueueRunner(RunnerBase):
    def __init__(self, model, devices, ds, threads, post_proc=None, tpu_kernel=None):
        super().__init__(model, devices, ds, threads, post_proc, tpu_kernel)
        self.query_samples = []
        self.result_dict = {}
        self.task_map = {}
        self.running = False
        self.cond = threading.Condition()
        self.in_queue = Queue(maxsize=10000)

    def start_run(self, result_dict, take_accuracy):
        super().start_run(result_dict, take_accuracy)
        self.running = True
        self.put_worker = threading.Thread(target=self.put_samples)
        self.get_worker = threading.Thread(target=self.handle_tasks)
        self.put_worker.start()
        self.get_worker.start()

    def put_samples(self):
        while self.running:
            with self.cond:
                while not self.query_samples:
                    if not self.running:
                        return
                    self.cond.wait()
                self.put(self.query_samples)
                self.query_samples.clear()

    def handle_tasks(self):
        """Worker thread."""
        while self.running:
            task_id, results, valid = self.runner.get()
            if task_id == 0:
                break
            qitem = self.task_map.pop(task_id)
            self.process_result(qitem, task_id, results, valid)

    def enqueue(self, query_samples):
        with self.cond:
            self.query_samples += query_samples
            self.cond.notify()

    def put(self, query_samples):
        idx = [q.index for q in query_samples]
        query_id = [q.id for q in query_samples]
        if len(query_samples) < self.max_batchsize:
            data, label = self.ds.get_samples(idx)
            task_id = self.runner.put(data)
            self.task_map[task_id] = Item(query_id, idx, data, label)
        else:
            bs = self.max_batchsize
            for i in trange(0, len(idx), bs, desc='running'):
                ie = i + bs
                data, label = self.ds.get_samples(idx[i:ie])
                task_id = self.runner.put(data)
                self.task_map[task_id] = Item(query_id[i:ie], idx[i:ie], data, label)

    def finish(self):
        with self.cond:
            self.running = False
            self.cond.notify()
        self.put_worker.join()
        self.runner.put()
        self.runner.show()
        self.get_worker.join()
