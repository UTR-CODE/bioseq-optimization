# 多进程优化与单GPU评分队列

[返回主文档](../README.md) · [串行参考流程](candidate-generation.md)

每种算法在独立CPU进程生成提案，通过有界队列发给同一个模型进程。服务把同类请求合并为GPU batch，再把结果送回各算法。梯度算法的soft前向和反向VJP也是队列请求；反向在服务中重算前向，不跨进程持有GPU计算图。模型必须为确定性的eval模式。

下面给出完整的组成部分，便于修改；不是要求只能运行一个隐藏的入口脚本。

## 1. 定义可导入的模型factory

`spawn`需要函数可以从模块导入。创建很小的工厂模块；权重仍由bioseq-models加载：

```python
# 保存为本次实验目录下的 my_model_factory.py
from bioseq_models import load_model

def make_utr_scorer():
    return load_model("optimus5prime", device="cuda", batch_size=512)
```

Enhancer factory可以在内部返回`MalinoisScorer(load_model("malinois",...), "K562")`；promoter factory内部加载`dream_legnet(variable_region_only=True)`后返回`PromoterScorer(raw)`。不要在主进程提前加载CUDA模型。

## 2. 提交11种优化算法

下面是驱动逻辑的全部内容；在有`__main__`保护的Python模块中运行。Notebook中优先使用前述串行步骤，或把这个驱动单元保存为模块再调用，避免交互函数无法被spawn反序列化。

```python
from sequence_optimization import random_seed_sequences
from sequence_optimization.workflows import generate_candidates_parallel
from my_model_factory import make_utr_scorer

if __name__ == "__main__":
    initial = random_seed_sequences(50, 100, seed=42)
    records, times = generate_candidates_parallel(
        make_utr_scorer, 50, initial, ["maximize"],
        task_names=("mrl",), output_dir="outputs/utr_parallel_run1",
        num_candidates=100, seed=42, diversity=5,
        max_batch_size=512, max_soft_batch_size=64, batch_wait_ms=5.,
    )
    assert len(records) == 1100
    print(times[["algorithm", "seconds", "queries", "stop_reason"]])
```

`task_names`必须与factory scorer输出列完全一致：enhancer有5列而不是3列，promoter/UTR各1列。默认启动11个优化进程和1个评分进程，每个优化进程限制Torch/Numba为1线程。在线程库导入之前也可设置：

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

## 3. 观察性能与故障

输出`gpu_batches.csv/json`记录每种请求行数、batch大小、跨worker合批数、服务时间；`<algorithm>.log`保存算法错误；`algorithm_summary.csv`包含各算法wall/cpu时间。并发walltime包括排队和共享GPU竞争，应同时报告总体walltime，不能直接称为独占GPU算法速度。

显存不足可降低hard/soft batch，长序列和ensemble尤其需要小batch。控制器检测失败worker或服务退出后清理自己启动的进程；不终止其他实验。无查询budget的运行可能很长；中途停止不属于已达到平台期，不能写成完整1,100条比较结果。

## 4. 与串行流程核对

两种流程使用相同`optimize`、种子和90%增益配置，输出表结构相同。但GPU不同batch形状可能产生小浮点误差，优化分叉后DNA不一定逐条一致。用one-hot hard/soft一致性、最终重新预测、数量、停止原因及距离约束验证，而非要求所有随机搜索轨迹逐步相同。
