# 11种算法生成1,100条候选

[返回主文档](../README.md) · [Enhancer](examples/enhancer.md) · [Promoter](examples/promoter.md) · [5′UTR](examples/5utr.md)

下面展示公共流程，`scorer`和`objective`由对应任务示例创建。每个算法从**相同的100条DNA**开始，单独优化、单独停止、单独保留100条。候选池收集完成以后才进行跨算法motif筛选。

## 1. 起始序列和停止配置

```python
from sequence_optimization import random_seed_sequences, SaturationConfig
from sequence_optimization.workflows import DESIGN_ALGORITHMS

length = 200                     # enhancer；promoter=80，5′UTR=50
initial = random_seed_sequences(length, 100, seed=42)
saturation = SaturationConfig(
    fractions=(0.9,),
    selection="fraction",
    maximum_queries=None,
)
assert len(DESIGN_ALGORITHMS) == 11
print(DESIGN_ALGORITHMS)
```

平台期基于候选archive的质量增益和patience窗口判定，不要求数学上达到全局最优。参数细节见[算法与停止规则](algorithms.md)。这里明确设置`selection="fraction"`，以使用90%增益档；它和默认的knee选择是两个可选策略。没有固定query budget，也不设置强制停止时长。

## 2. 显式优化循环

以下是核心过程，适合研究者自行修改。先从任务文档创建`scorer`和`objective`，再执行：

```python
from pathlib import Path
import time
import pandas as pd
from sequence_optimization import optimize

root = Path("outputs/explicit_loop_run1")
root.mkdir(parents=True, exist_ok=False)
frames, timing = [], []
for name in DESIGN_ALGORITHMS:
    started = time.perf_counter()
    result = optimize(
        scorer, length, initial, objective, 100,
        algorithm=name, seed=42, diversity=5,
        saturation=saturation, count_repeated=True,
    )
    seconds = time.perf_counter() - started
    assert result.stop_reason == "plateau"
    retained = result.at(0.9)
    assert len(retained.sequences) == len(set(retained.sequences)) == 100
    frame = pd.DataFrame({
        "record_id": [f"{name}_{i+1:03}" for i in range(100)],
        "sequence": retained.sequences,
        "algorithm": name,
        "score": retained.scores,
    })
    for j, task in enumerate(scorer.task_names):
        frame[f"activity_{task}"] = retained.activities[:, j]
    frame.to_csv(root / f"{name}.csv", index=False)
    frames.append(frame)
    timing.append({"algorithm": name, "seconds": seconds,
                   "queries": result.total_query_count,
                   "stop_reason": result.stop_reason, "candidates": len(frame)})

records = pd.concat(frames, ignore_index=True)
times = pd.DataFrame(timing)
assert len(records) == 1100
records.to_csv(root / "all_candidates.csv", index=False)
times.to_csv(root / "algorithm_summary.csv", index=False)
```

`diversity=5`是**每个算法输出内部**的最小Hamming距离，与后续motif筛选不同。不能把同一个算法返回不足100条的结果补成重复记录来满足1,100行。

## 3. 可复用接口

生产运行可以调用同一逻辑的封装，它另外保存种子、配置、轨迹、FASTA，并重新预测返回的候选做一致性检查：

```python
from sequence_optimization.workflows import generate_candidates

records, times = generate_candidates(
    scorer, length, initial, objective,
    output_dir="outputs/candidate_pool_run1",
    algorithms=DESIGN_ALGORITHMS, num_candidates=100,
    saturation=saturation, diversity=5, seed=42,
)
print(times[["algorithm", "seconds", "cpu_seconds", "queries", "stop_reason"]])
```

`seconds`包括该算法运行中的模型查询，最终一致性重评分在计时之外。多进程版本还包含队列等待，因此不能把并发耗时当成算法独占GPU的速度排名；见[队列示例](parallel-scoring.md)。

## 4. 去重并保留来源

```python
from sequence_optimization.workflows import deduplicate_candidates, write_fasta

unique, provenance = deduplicate_candidates(records)
assert len(provenance) == 1100
assert unique.sequence.nunique() == len(unique)
unique.to_csv("outputs/unique_candidates.csv", index=False)
provenance.to_csv("outputs/record_provenance.csv", index=False)
write_fasta("outputs/unique_candidates.fa", unique.sequence_id, unique.sequence)
print(len(unique), "unique DNA sequences")
```

`record_id`标识一个算法的候选记录；`sequence_id`由DNA的SHA256确定；`algorithms`记录每条唯一DNA出现于哪些算法。FIMO只扫描唯一DNA，然后按来源映射回全部1,100条记录。不要把CSV行号作为永久的序列标识。

## 5. 可选Random Search参考

Random不加入主候选池；固定生成200,000条新随机序列、评分、挑出前100条，不能声称它也达到平台期。

```python
import numpy as np
from sequence_optimization.api import as_score_batch

pool = random_seed_sequences(length, 200_000, seed=43)
predictions = as_score_batch(scorer.score(pool)).scores
# 单输出模型：第一列；Malinois：两条target-offtarget margin的最小值。
random_scores = predictions[:, 0] if predictions.shape[1] == 1 else predictions[:, 3:].min(axis=1)
chosen = np.argsort(-random_scores, kind="stable")[:100]
reference = pd.DataFrame({"sequence": pool[chosen], "score": random_scores[chosen]})
reference.to_csv("outputs/random_reference_top100.csv", index=False)
```

以上标量化只针对本文三个示例的输出布局；换模型需要明确自己的objective。
