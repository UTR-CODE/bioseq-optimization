"""Maintain step-by-step notebooks from the documented task examples."""
from pathlib import Path
import ast,json,re
ROOT=Path(__file__).resolve().parents[1]
SETUP='''from pathlib import Path
import os
if Path.cwd().name == "examples" and (Path.cwd().parent / "pyproject.toml").exists():
    os.chdir(Path.cwd().parent)
print("Working directory:", Path.cwd())
'''

def make(text,name):
    cells=[dict(cell_type='code',metadata={},source=SETUP.splitlines(True),execution_count=None,outputs=[])]
    pos=0
    for match in re.finditer(r'```python\n(.*?)\n```',text,re.S):
        if match.start()>pos:cells.append(dict(cell_type='markdown',metadata={},source=text[pos:match.start()].splitlines(True)))
        code=match.group(1)+'\n';ast.parse(code)
        cells.append(dict(cell_type='code',metadata={},source=code.splitlines(True),execution_count=None,outputs=[]));pos=match.end()
    if text[pos:].strip():cells.append(dict(cell_type='markdown',metadata={},source=text[pos:].splitlines(True)))
    for i,cell in enumerate(cells):
        cell['id']=f'{name}-{i:03}'
        if cell['cell_type']=='markdown':
            s=''.join(cell['source']).replace('../../README.md','../README.md').replace('../../examples/','')
            for doc in ['candidate-generation','parallel-scoring','motif-discovery','motif-diversity']:
                s=s.replace(f'../{doc}.md',f'../docs/{doc}.md')
            cell['source']=s.splitlines(True)
    return dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.10')),nbformat=4,nbformat_minor=5)

for name in ['enhancer','promoter','5utr']:
    text=(ROOT/f'docs/examples/{name}.md').read_text()
    if name=='enhancer':
        a=text.index('## 扩展到小鼠组织');b=text.index('## 7.',a)
        text=text[:a]+'可选小鼠模型加载示例见[完整文档](../docs/examples/enhancer.md)。\n\n'+text[b:]
    (ROOT/f'examples/{name}.ipynb').write_text(json.dumps(make(text,name),ensure_ascii=False,indent=1)+'\n')
text=(ROOT/'docs/motif-diversity.md').read_text().split('## 6. 无GPU、无需FIMO安装的真实参考数据')[1]
text='# K562真实参考输入：免GPU重建Top10\n\n'+text
text+='''\n\n比较已保存的参考序列及顺序：\n\n```python
expected = pd.read_csv(DATA / "k562_expected_top10.csv")
assert selected.sequence.tolist() == expected.sequence.tolist()
assert selected.sequence_id.tolist() == expected.sequence_id.tolist()
print("PASS: exact Top10 DNA and selection order reproduced")
```\n'''
(ROOT/'examples/motif_diversity.ipynb').write_text(json.dumps(make(text,'motif-diversity'),ensure_ascii=False,indent=1)+'\n')
print('Generated 3 task notebooks and 1 offline motif-diversity notebook')
