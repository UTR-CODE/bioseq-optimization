# 仓库工作规则

## 文档

- README 和用户文档面向使用者，只写使用方法、参数含义、示例和必要限制。
- 不要把开发过程、接口迁移、兼容性验证、测试通过数量、工作总结或内部提醒写进用户文档。这些内容写在 Git commit message 中。
- 不用“现在支持”“无需再包装”“不需要定义某个对象”等措辞解释内部改动或回应历史对话。用代码示例表达当前用法。
- 示例链接可以用简短的链接标题呈现，不添加兼容性说明或工作汇报式的引导句。
- 自定义模型封装说明放在独立文档，README 末尾保留链接。
- README 的优化示例统一使用遗传算法；可用算法表放在示例之前。其他算法的详细用法放在独立文档。Multi-task 主示例使用三输出 CNN。
- README 前部保留 single-task、multi-task、multiple independent models 三个 custom CNN 示例。具体预训练模型任务和每个算法/模型的参数各放独立页面，由索引和示例下方链接访问。
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
