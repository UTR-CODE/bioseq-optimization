# 验证、数据来源与适用范围

[返回主文档](../README.md)

## 自动检查

```bash
python -m pytest
python tools/check_docs.py
python -m pip wheel . --no-deps --no-build-isolation -w /tmp/bioseq-optimization-wheel
```

测试覆盖优化器契约、质量与探索、90%增益/平台期、ISM lineage退出、可选C构造、motif相似性与侧翼、Soft-Jaccard、Hungarian匹配、无motif证据、record加权支持、完整11×100记录和去重来源，以及跨进程评分与梯度路由。候选生成既有11算法的数量/provenance契约测试，也有实际执行到平台期的串行和双优化进程测试；不会在普通CI里重跑三个任务全部生产规模。

发布前已在当前环境完成含C扩展的安装包测试：501项通过，1项按设计跳过（Random Search不使用平台期停止规则）；另执行免GPU Notebook全部单元，重现原Top10的DNA和顺序。并用真实checkpoint检查以下模型：

| 模型 | 检查 |
|---|---|
| Optimus5Prime | 4条50 nt序列，字符串与one-hot soft相同，输入梯度有限且非零 |
| Malinois | 4条200 nt序列，完整MPRA上下文一致，hard/soft及梯度检查 |
| DREAM LegNet | 4条80 nt插入片段，hard/soft一致，110 nt完整构建与固定上下文包装一致 |

三者此次hard/soft最大差均为0；完整promoter构建差也为0。硬件、库版本和batch形状改变后可有小浮点误差，示例使用容差检查。当前模型检查记录见[model_validation.json](../examples/data/model_validation.json)。

## 参考数据复现

[K562 reference inputs](../examples/data/README.md)包含1,100来源记录、1,084条唯一DNA、11个catalog motif、12,164个过滤后的物理位点。按[motif教程](motif-diversity.md#6-无gpu无需fimo安装的真实参考数据)执行，可以重现原Top10的DNA及顺序；参考文件见[k562_expected_top10.csv](../examples/data/k562_expected_top10.csv)。稳定ID重建不会改变DNA、预测和motif位点。输入SHA256见[manifest.json](../examples/data/manifest.json)。

这些数据来自此前已完成的Malinois K562运行；本次整理未重新训练模型，也未重跑所有三个任务的全规模优化。新的支持门槛/holdout motif发现教程是可执行方法说明，不应被理解为已经在promoter和UTR上发现并实验验证同样的motif。FIMO扫描中保留原始完整catalog的q值，打包保留的11模型不会触发重新计算。

## 生物学解释边界

Enhancer的motif多样性以本次catalog为参照，无法覆盖的机制不能因“无命中”被称为全新机制。PWM数据库注释是相似性，不证明真实TF结合或目标特异作用。

Promoter与UTR的基础示例选择DNA差异，不声称已经完成motif机制多样性验证。UTR可能涉及RNA结构、uAUG和完整转录本上下文；酵母promoter模型不能被解释为人类组织特异性模型。小鼠模型对应E11.5胚胎，不代表成年组织。

所有输出是模型预测，最终实验应安排阳性/阴性对照、重复与模型之外的验证。高模型分数本身不是湿实验结果。
