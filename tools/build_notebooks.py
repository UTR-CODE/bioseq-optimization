"""Generate task notebooks from the Markdown workflows."""
from pathlib import Path
import ast
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def notebook(text, name):
    cells = []
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


def write_notebook(path, current):
    if path.exists():
        previous = json.loads(path.read_text())
        old_code = [cell for cell in previous['cells'] if cell['cell_type'] == 'code']
        new_code = [cell for cell in current['cells'] if cell['cell_type'] == 'code']
        old_source = [''.join(cell['source']).rstrip() for cell in old_code]
        new_source = [''.join(cell['source']).rstrip() for cell in new_code]
        if old_source == new_source:
            for old, new in zip(old_code, new_code):
                new['execution_count'] = old['execution_count']
                new['outputs'] = old['outputs']
    path.write_text(json.dumps(current, ensure_ascii=False, indent=1) + '\n')


for source in sorted((ROOT / 'docs/examples').glob('*.md')):
    name = source.stem
    write_notebook(ROOT / f'examples/{name}.ipynb', notebook(source.read_text(), name))
text = (ROOT / 'docs/motif-diversity.md').read_text().split('## 6. Reference-data example')[1].split('## Outputs')[0]
text = '# Reference motif-diversity selection\n\n' + text.replace('Path("examples/data")', 'Path("data")')
write_notebook(ROOT / 'examples/motif_diversity.ipynb', notebook(text, 'motif-diversity'))
print('Generated workflow notebooks')
