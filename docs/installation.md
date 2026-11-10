# 安装与模型接口

[返回主文档](../README.md) · 下一步：[候选生成](candidate-generation.md)

建议Python 3.10以上，独立虚拟环境。先按服务器的CUDA版本安装PyTorch，再安装本项目：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[analysis,motifs,dev]'
python -m pip install 'bioseq-models @ git+https://github.com/UTR-CODE/bioseq-models.git'
```

模型调用来自 [bioseq-models/bioseq_models](https://github.com/UTR-CODE/bioseq-models/tree/main/bioseq_models)。第一次加载模型可能下载checkpoint；缓存由该包管理。版本和checkpoint标识应随实验记录一起保存。

```python
import importlib.metadata as metadata
import torch
from bioseq_models import load_model

print(metadata.version("bioseq-models"), metadata.version("bioseq-optimization"))
print(torch.__version__, torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("optimus5prime", device=device, batch_size=512)
print(model.task_names)
```

`score(sequences)`接受DNA字符串，返回`[N, tasks]`预测值。梯度优化还需要`score_soft(probabilities)`：输入`[N,L,4]`、ACGT通道，返回`(预测Tensor, uncertainty)`。不是每个模型adapter都原生支持soft输入，因此本项目只为示例补充**固定上下文与编码转换**，仍调用`load_model`得到的网络；见[models.py](../sequence_optimization/models.py)。

| 示例 | 模型ID | 优化区域 | 预测值 |
|---|---|---:|---|
| 人细胞enhancer | `malinois` | 200 nt | K562、HepG2、SKNSH活性；可构建特异性差值 |
| 酵母promoter | `dream_legnet` | 80 nt插入片段 | 预期FACS bin分数，约0–17的标签尺度 |
| 5′UTR | `optimus5prime` | 50 nt | mean ribosome load（MRL） |

模型间的分数不在同一尺度，不应直接比较，`min_score=20`只用于本仓库Malinois参考示例，不能照搬到MRL或概率模型。

## 可选外部motif工具

例如使用独立conda环境：

```bash
conda install -c conda-forge -c bioconda meme r-base
fimo --version
streme --version
tomtom --version
Rscript --version
```

如果MEME工具不在PATH，可向`scan_fimo(..., executable="/path/to/fimo")`提供路径。tangermeme作为Python可选依赖安装；需要其ISM以外的DeepLIFTShap功能时，也请查阅其[官方文档](https://tangermeme.readthedocs.io/)。本项目不下载或打包第三方motif数据库；使用JASPAR等数据库时记录版本、物种、来源和许可。

## 安装检查

```bash
python -m pytest
```

单元测试不需要checkpoint或GPU。真实模型检查与验证范围见[验证文档](validation.md)。
