import os
import sys
import argparse
import time
import numpy as np
import dataloder
from dataloder import get_dataloader
import random
from yolov5_runner import QueueRunner, RunnerBase

sys.path.insert(0, os.getcwd())

SUPPORTED_DATASETS = {
    "coco-640":
        (dataloder.coco_loader, dataloder.pre_process_coco_yolov5, dataloder.PostProcessCocoYolo(True, 0.05),
         {"image_size": [640, 640, 3], "use_label_map": True}),
    "coco-544":
        (dataloder.coco_loader, dataloder.pre_process_coco_yolov5, dataloder.PostProcessCocoYolo(True, 0.05),
         {"image_size": [544, 544, 3], "use_label_map": True}),
    "coco-custom":
        (dataloder.coco_loader, dataloder.pre_process_coco_yolov5, dataloder.PostProcessCocoYolo(True, 0.05),
         {"image_size": [128, 1024, 3], "use_label_map": True})
}

last_timeing = []


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/mnt/onager/source/datasets/coco", help="data file path")
    parser.add_argument("--dataset_name", choices=SUPPORTED_DATASETS.keys(),
                        default="coco-640", help="dataset config name")
    parser.add_argument("--scenario", choices=["SingleStream", "Offline"], default="Offline", help="Scenario")
    parser.add_argument("--accuracy", action="store_true", help="enable accuracy pass")
    parser.add_argument("--count", type=int, default=50, help="Maximum number of examples to consider")
    parser.add_argument("--model", type=str, help="bmodel path")
    parser.add_argument("--cache_path", default=None, help="pickle path")
    parser.add_argument("--out_file", default='./result/predictions.json', help='output json path')
    parser.add_argument("--data-format", choices=["NCHW", "NHWC"], help="data format")
    parser.add_argument("--threads", default=os.cpu_count(), type=int, help="threads")
    parser.add_argument("--save_result", action="store_true", help="save detected results")
    parser.add_argument("--output", default="output", help="test results")
    parser.add_argument("--performance-count", default=100, type=int,
                        help="performance sample count while acc mode & offline")
    parser.add_argument("--seed", default=123, type=int, help="seed in all random function")
    parser.add_argument("--tpu-kernel", type=str, help="Utilize tpu kernel as post-process")

    args = parser.parse_args()
    return args


class QuerySample():
    def __init__(self, index, query_id):
        self.index = index  # Identify data
        self.id = query_id  # Identify query


# control the timing to send query
def load_gen(config, runner, dl):
    count = config['total_count']
    query_id_counter = max(100000, count)
    random.seed(config['seed'])
    if config['scenario'] == 'SingleStream':
        return
    elif config['scenario'] == 'Offline':
        all_list = [i for i in range(len(dl))]
        queries = []
        import math
        performance_count = min(len(dl), config['performance_count'])
        if config['accuracy'] is not None:
            random.shuffle(all_list)
            for i in range(math.ceil(count / performance_count)):
                queries.clear()
                end = min((i + 1) * performance_count, len(dl))
                performance_list = []
                for n in all_list[i * performance_count: end]:
                    performance_list.append(n)
                    runner.waiting_queries[query_id_counter] = ''
                    queries.append(QuerySample(n, query_id_counter))
                    query_id_counter += 1
                dl.load_query_samples(performance_list)
                runner.enqueue(queries)
                runner.wait_for_response()
                dl.unload_query_samples()
        else:
            for i in range(math.ceil(count / performance_count)):
                queries.clear()
                end = min((i + 1) * performance_count, count)
                performance_list = []
                for _ in range(i * performance_count, end):
                    idx = random.choices(all_list)
                    performance_list.append(idx)
                    runner.waiting_queries[query_id_counter] = ''
                    queries.append(QuerySample(idx, query_id_counter))
                    query_id_counter += 1
                dl.load_query_samples(performance_list)
                runner.enqueue(queries)
                runner.wait_for_response()
                dl.unload_query_samples()
    else:
        return
    runner.finish()


def get_runner(args, ds, post_proc):
    if args.scenario == 'Offline':
        return QueueRunner(args.model, ds, args.threads, post_proc=post_proc, tpu_kernel=args.tpu_kernel)
    elif args.scenario == 'SingleStream':
        return RunnerBase(args.model, ds, args.threads, post_proc=post_proc, tpu_kernel=args.tpu_kernel)


def add_results(final_results, name, result_dict, result_list, took, show_accuracy=False):
    percentiles = [50., 80., 90., 95., 99., 99.9]
    buckets = np.percentile(result_list, percentiles).tolist()
    buckets_str = ",".join(["{}:{:.4f}".format(p, b) for p, b in zip(percentiles, buckets)])

    if result_dict["total"] == 0:
        result_dict["total"] = len(result_list)

    # this is what we record for each run
    result = {
        "took": took,
        "mean": np.mean(result_list),
        "percentiles": {str(k): v for k, v in zip(percentiles, buckets)},
        "qps": len(result_list) / took,
        "count": len(result_list),
        "good_items": result_dict["good"],
        "total_items": result_dict["total"],
    }
    acc_str = ""
    if show_accuracy:
        result["accuracy"] = 100. * result_dict["good"] / result_dict["total"]
        acc_str = ", acc={:.3f}%".format(result["accuracy"])
        if "mAP" in result_dict:
            result["mAP"] = 100. * result_dict["mAP"]
            acc_str += ", mAP={:.3f}%".format(result["mAP"])

    # add the result to the result dict
    final_results[name] = result

    # to stdout
    print("{} qps={:.2f}, mean={:.4f}, time={:.3f}{}, queries={}, tiles={}".format(
        name, result["qps"], result["mean"], took, acc_str,
        len(result_list), buckets_str))


def main():
    global last_timeing
    if not os.path.exists("result"):
        os.makedirs("result")
    args = get_args()
    final_results = {
        "runtime": 'sgInfer',
        "version": 'version',
        "time": int(time.time()),
        "cmdline": str(args),
    }
    config = {
        "accuracy": args.accuracy,
        "total_count": args.count,
        "performance_count": args.performance_count if args.performance_count else args.count,
        "scenario": args.scenario,
        "seed": args.seed
    }
    image_format = args.data_format if args.data_format else 'NCHW'
    wanted_dataset, pre_proc, post_proc, kwargs = SUPPORTED_DATASETS[args.dataset_name]

    dl = get_dataloader(count=args.count,
                        input_file=args.data,
                        cache_path=args.cache_path,
                        name=args.dataset_name,
                        image_format=image_format,
                        pre_process=pre_proc,
                        use_cache=args.cache_path,
                        **kwargs)
    runner = get_runner(args, dl, post_proc)
    dl.model_info = runner.info

    # warmup
    sample_ids = [0]
    dl.load_query_samples(sample_ids)
    for _ in range(1):
        # img, label = dl.get_samples(sample_ids)
        shape = runner.info[list(runner.info.keys())[0]]['shape']
        img = np.zeros(shape, dtype=np.float32)
        img = img.reshape([1, 3, 640, 640])
        runner.runner.put(img)
        task_id, results, valid = runner.runner.get()
        print(1)
    dl.unload_query_samples(None)
    print('Warnup ok')

    result_dict = {"good": 0, "total": 0}
    runner.start_run(result_dict, args.accuracy)
    load_gen(config, runner, dl)
    dl.unload_query_samples(None)

    if not last_timeing:
        last_timeing = runner.result_timing
    if args.output:
        output_dir = os.path.abspath(args.output)
        os.makedirs(output_dir, exist_ok=True)
        os.chdir(output_dir)
    if args.accuracy and args.output:
        # Do evaluation
        post_proc.finalize(result_dict, dl, output_dir=output_dir, save_result=args.save_result)
    add_results(final_results, "{}".format(args.scenario),
                result_dict, last_timeing, time.time() - dl.last_loaded, args.accuracy)

    print("Done!")


if __name__ == "__main__":
    main()
