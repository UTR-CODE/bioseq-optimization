# Quick Start Guide: Multi-Algorithm Optimization Framework

## 立即开始

### 1. 安装包
```bash
cd /mnt/data/home/hexin/MPRA_benchmark
python -m pip install -e .
```

### 2. 快速测试（可选）
```bash
# 运行小规模测试验证框架工作正常
python sequence_optimization/examples/test_suite_framework.py
```

### 3. 运行示例任务

#### 选项 A: 运行单个任务
```bash
# 任务1: Optimus 5-Prime MRL (单任务优化)
python sequence_optimization/examples/01_optimus_mrl_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42

# 任务2: DeepSTARR 增强子 (多任务优化)
python sequence_optimization/examples/02_deepstarr_multitask_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42

# 任务3: Agarwal lentiMPRA (多模型优化)
python sequence_optimization/examples/03_agarwal_multimodel_suite.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42
```

#### 选项 B: 一次运行所有任务
```bash
python sequence_optimization/examples/run_all_suites.py \
    --device cuda \
    --initial-count 100 \
    --num-candidates 50 \
    --budget 5000 \
    --seed 42
```

### 4. 查看结果
```bash
# 查看生成的结果目录
ls -lh results/

# 查看汇总统计
head -20 results/01_optimus_mrl_suite/summary.csv

# 查看所有候选序列
head -50 results/01_optimus_mrl_suite/candidates.csv
```

## 主要参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--algorithms` | 全部13个 | 指定运行哪些算法 |
| `--initial-count` | 100 | 初始种群大小 |
| `--num-candidates` | 50 | 每个算法返回的候选数量 |
| `--budget` | 5000 | 每个算法的查询预算 |
| `--device` | cuda | 推理设备 (cuda/cpu) |
| `--seed` | 42 | 随机种子 |

## Python API 使用

### 基础用法
```python
from bioseq_models import load_model
from sequence_optimization import (
    CandidatePool,
    optimize_algorithms,
    random_seed_sequences,
)

# 加载模型
model = load_model("optimus5prime", device="cuda")

# 生成初始序列
initial = random_seed_sequences(length=50, count=100, seed=42)

# 运行优化套件
suite = optimize_algorithms(
    model,
    algorithms=["genetic_algorithm", "basesearch", "beam_search"],
    length=50,
    initial=initial,
    objective=["maximize"],
    num_candidates=50,
    budget=5000,
    seed=42,
)

# 创建统一候选池
pool = CandidatePool.from_suite_result(suite)

# 分析结果
print(pool)  # 显示统计信息
df = pool.to_frame()  # 转换为 DataFrame
convergence = pool.get_convergence_summary()  # 收敛分析
unique = pool.deduplicate(keep='best')  # 去重
top_100 = pool.top_n_by_score(100)  # 选择 top-N
```

### 分析候选池
```python
# 查看哪些序列被多个算法找到
convergence = pool.get_convergence_summary()
multi_algo = convergence[convergence['algorithm_count'] > 1]
print(multi_algo)

# 按算法比较最佳分数
df = pool.to_frame()
best_by_algo = df.groupby('algorithm')['score'].max()
print(best_by_algo)

# 选择不同策略的代表序列
evolutionary = df[df['algorithm'].isin(['genetic_algorithm', 'adalead'])]
systematic = df[df['algorithm'].isin(['ism_greedy', 'beam_search'])]
```

## 预期运行时间

**在 RTX 4090 GPU 上的预期运行时间：**

| 任务 | 序列长度 | 13个算法 | 预期时间 |
|------|----------|----------|----------|
| Optimus MRL | 50 nt | 100 初始, 50 候选, 5000 预算 | ~5-10 分钟 |
| DeepSTARR | 249 nt | 100 初始, 50 候选, 5000 预算 | ~10-15 分钟 |
| Agarwal (3模型) | 230 nt | 100 初始, 50 候选, 5000 预算 | ~15-25 分钟 |

**在 CPU 上运行（测试/调试）：**
```bash
# 使用更小的参数
python sequence_optimization/examples/01_optimus_mrl_suite.py \
    --device cpu \
    --initial-count 20 \
    --num-candidates 10 \
    --budget 500
```

## 输出文件结构

每个任务生成一个结果目录：

```
results/01_optimus_mrl_suite/
├── config.json              # 运行配置（可复现参数）
├── candidates.csv          # 所有算法的所有候选（统一池）
├── summary.csv             # 每个算法的性能统计
├── initial.csv             # 共享的初始种群
├── genetic_algorithm.csv   # 单个算法的结果
├── basesearch.csv
├── beam_search.csv
├── ... (全部13个算法)
└── ledidi.csv
```

## 常见问题

### GPU 内存不足
```bash
# 减小批次大小
--model-batch-size 2048
```

### 运行特定算法
```bash
# 只运行3个算法
--algorithms genetic_algorithm basesearch beam_search
```

### 调整输出数量
```bash
# 每个算法返回更多候选
--num-candidates 100

# 增加搜索预算
--budget 10000
```

## 下一步

生成的候选池现在可以用于：

1. **质量控制分析** (即将实现)
   - 离群点检测
   - 不确定性量化
   
2. **解决方案家族发现** (即将实现)
   - 序列聚类
   - Motif 提取
   
3. **代表性选择** (即将实现)
   - 多样性采样
   - 实验验证推荐

## 帮助与文档

- 完整框架指南: `docs/multi_algorithm_framework.md`
- 详细示例文档: `sequence_optimization/examples/SUITE_EXAMPLES.md`
- 实现总结: `docs/IMPLEMENTATION_SUMMARY.md`
- 项目交接文档: `HANDOFF.md`

## 获取帮助

```bash
# 查看脚本帮助
python sequence_optimization/examples/01_optimus_mrl_suite.py --help
python sequence_optimization/examples/run_all_suites.py --help
```

祝优化顺利！🚀
