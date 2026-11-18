"""Generate task notebooks from the Markdown workflows."""
from pathlib import Path
import ast
import json
import re

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import os
if Path.cwd().name == "examples" and (Path.cwd().parent / "pyproject.toml").exists():
    os.chdir(Path.cwd().parent)
print("Working directory:", Path.cwd())
'''


def notebook(text, name):
    cells = [dict(cell_type='code', metadata={}, source=SETUP.splitlines(True), execution_count=None, outputs=[])]
    pos = 0
    for match in re.finditer(r'```python\n(.*?)\n```', text, re.S):
        if match.start() > pos:
            cells.append(dict(cell_type='markdown', metadata={}, source=text[pos:match.start()].splitlines(True)))
        code = match.group(1) + '\n'
        ast.parse(code)
        cells.append(dict(cell_type='code', metadata={}, source=code.splitlines(True), execution_count=None, outputs=[]))
        pos = match.end()
    if text[pos:].strip():
        cells.append(dict(cell_type='markdown', metadata={}, source=text[pos:].splitlines(True)))
    for i, cell in enumerate(cells):
        cell['id'] = f'{name}-{i:03}'
        if cell['cell_type'] == 'markdown':
            value = ''.join(cell['source']).replace('../../README.md', '../README.md').replace('../../examples/', '')
            value = re.sub(r'\.\./([a-z-]+)\.md', r'../docs/\1.md', value)
            value = re.sub(r'\]\(([a-z0-9-]+)\.md\)', r'](../docs/examples/\1.md)', value)
            cell['source'] = value.splitlines(True)
    return dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.10')), nbformat=4, nbformat_minor=5)


for name in ['enhancer', 'promoter', '5utr', 'single-task', 'multi-task', 'multiple-models', 'saluki-egfp']:
    text = (ROOT / f'docs/examples/{name}.md').read_text()
    (ROOT / f'examples/{name}.ipynb').write_text(json.dumps(notebook(text, name), ensure_ascii=False, indent=1) + '\n')
text = (ROOT / 'docs/motif-diversity.md').read_text().split('## 6. Reference-data example')[1].split('## Outputs')[0]
text = '# Reference motif-diversity selection\n\n' + text
(ROOT / 'examples/motif_diversity.ipynb').write_text(json.dumps(notebook(text, 'motif-diversity'), ensure_ascii=False, indent=1) + '\n')
print('Generated eight English workflow notebooks')
