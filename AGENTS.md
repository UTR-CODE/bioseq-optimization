# 仓库工作规则

## 文档

- README 和用户文档面向使用者，只写使用方法、参数含义、示例和必要限制。
- 不要把开发过程、接口迁移、兼容性验证、测试通过数量、工作总结或内部提醒写进用户文档。这些内容写在 Git commit message 中。
- 不用“现在支持”“无需再包装”“不需要定义某个对象”等措辞解释内部改动或回应历史对话。用代码示例表达当前用法。
- 示例链接可以用简短的链接标题呈现，不添加兼容性说明或工作汇报式的引导句。
- 自定义模型封装说明放在独立文档，README 末尾保留链接。
- README 的优化示例统一使用遗传算法；可用算法表放在示例之前。其他算法的详细用法放在独立文档。Multi-task 主示例使用三输出 CNN。
- README 前部保留 single-task、multi-task、multiple independent models 三个 custom CNN 示例。具体预训练模型任务和每个算法/模型的参数各放独立页面，由索引和示例下方链接访问。
- 模型任务页直接展示模型加载、目标和各 optimizer 的独立调用，不用循环或任务辅助函数隐藏用法。保留 initial、budget 和饱和筛选的完整说明；只展示满足输入和约束要求的算法，注明其余算法的限制。
- 饱和停止通过 optimizer 的 `saturation=SaturationConfig(...)` 参数使用，不提供独立的饱和优化函数；此时不能同时传 `budget`。普通示例在代码块内用 `random_seed_sequences` 构造 initial；CDS 优化使用明确列出的同义 CDS 初始序列，不做随机初始化。主示例展示 `to_frame()` 中的预测结果。
- CDS 优化必须校验 `ATG` 起始密码子和 `TAA`、`TAG` 或 `TGA` 终止密码子；所有初始和优化所得序列必须与参考 CDS 编码相同的氨基酸序列。
- 所有 optimizer 的 `constraint` 参数接受一个函数或多个函数；每个函数独立返回 `[0, 1]` 通过概率，只有所有函数筛选结果的交集进入模型评分和下一轮候选。
- 修改文档时检查全文，删除同类表述；不要只修复用户指出的一句，也不要新增“本次清理了什么”的文档记录。

用户明确指出以下原句属于内部说明，不得再次放入 README 或用户指南：

> No wrapper is required for Optimus with any of the nine black-box algorithms. The runnable version is sequence_optimization/examples/external_model.py.

其他禁止放入用户文档的例子：

- “directly; no sequence-space object is needed”
- “之前的接口已改成……”
- “全部测试已通过，本次新增……”

## 提交

- 完成修改后执行与改动相称的检查，并提交、推送到 GitHub。
- 纯文案修改不需要重复运行模型测试；检查 diff 和文档一致性即可。
