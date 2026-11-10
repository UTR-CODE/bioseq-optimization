# 算法与停止规则

[返回主文档](../README.md) · [候选生成](candidate-generation.md)

主流程的`DESIGN_ALGORITHMS`固定为以下11种；注册表另有`random`，因此`available_algorithms()`会返回12种，不能直接用注册表全部算法生成所谓的11算法候选池。

| API名称 | 方法 | 需要输入梯度 |
|---|---|---|
| `adalead` | 以质量为主、保留探索的进化搜索 | 否 |
| `basesearch` | 单碱基提案搜索 | 否 |
| `beam_search` | beam搜索 | 否 |
| `cross_entropy_method` | 概率分布更新 | 否 |
| `den` | 生成器优化 | 是 |
| `fast_seqprop` | FastSeqProp | 是 |
| `genetic_algorithm` | 遗传算法 | 否 |
| `ism_greedy` | 单碱基饱和突变的贪心搜索 | 否 |
| `ledidi` | 可微离散序列优化 | 是 |
| `simulated_annealing` | 模拟退火 | 否 |
| `vanilla_activation_maximization` | activation maximization | 是 |

名称是本仓库实现的API，不表示与每篇原始论文的默认超参数完全相同。具体实现见[algorithms.py](../sequence_optimization/algorithms.py)、[advanced.py](../sequence_optimization/advanced.py)。AdaLead保留质量和探索策略；AM不因本次整理而修改。

## 饱和与90%增益

```python
from sequence_optimization import SaturationConfig
saturation = SaturationConfig(
    fractions=(0.9,), selection="fraction", maximum_queries=None,
    material_improvement=0.01,
    relative_material_improvement=0.001,
)
```

这是本次示例明确采用的配置。基础patience窗口随起始种群/保留候选数缩放，未指定时分别约为10N和40N查询；算法还考虑提案batch、轮次patience等条件，避免大邻域尚未完成就停止。`stop_reason="plateau"`表示当前控制器判定继续增益过小，不表示证明了全局最优。返回数量、门槛和停止原因均写入结果。

90%增益档以搜索观测的初始archive质量和最终archive质量为参照，取得达到相应增益水平的候选archive；它不是“每条候选的分数至少等于最高分×0.9”。因此仍需为实验候选设置**任务尺度下的质量门槛**。可从`trajectory.json`检查曲线；`result.at(.9)`取增益档；knee可作为另一种研究配置，不能把两者混用后仍标成相同实验。

ISM会对仍活跃的lineage生成单碱基邻域，单条饱和后可以退出，不必每轮都查询100×L×3条。C扩展加速序列构造等操作，不改变预测模型和优化目标。查询次数记录提案成本；缓存复用后的实际模型行数和GPU批次另有统计，不能混称。
