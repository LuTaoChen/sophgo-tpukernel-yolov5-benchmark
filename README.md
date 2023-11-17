# Yolov5 Benchmark mlperf
-----------------------------------

测试yolov5 后处理tpu-kernel算子的速度和精度

## Dataset
[COCO 2017](http://cocodataset.org)

## 依赖

安装支持tpu-kernel runtime的[tpu-perf dev](https://github.com/LuTaoChen/tpu-perf/tree/dev)分支

```shell
git clone git@github.com:LuTaoChen/tpu-perf.git
cd tpu-perf
git fetch --all
git checkout dev
```

1、按照README.md编译libpipeline.so

2、安装 `pip install tpu_perf-1.0.11-py3-none-manylinux2014_x86_64.whl`

## 模型
```
# model I/O shapes like
# [ 1 3 H W ] for input
# [ 1 255 80 80 ] 
# [ 1 255 40 40 ]
# [ 1 255 20 20 ] for outputs
```




## Inference
For example
```shell
export TPUKERNEL_DIR=/path/to/dir/libbm1684x_kernel_module.so
python run.py --model=yolov5.bmodel --dataset_name="coco-640" --accuracy
```

参数说明：

--accuracy: 是否打开计算精度

--count: 运行的样本数

--model: 模型路径

--dataset_name: 数据集配置
