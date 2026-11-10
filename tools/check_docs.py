"""Check local Markdown targets and Python/Notebook syntax without running searches."""
import ast
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
errors=[];blocks=0
for p in [ROOT/'README.md',*sorted((ROOT/'docs').rglob('*.md')),*sorted((ROOT/'examples').rglob('*.md'))]:
    text=p.read_text()
    if any("\u4e00" <= ch <= "\u9fff" for ch in text):
        errors.append(f"{p.relative_to(ROOT)}: expected English documentation")
    for target in re.findall(r'\]\(([^)]+)\)',text):
        if '://' in target or target.startswith('#'):continue
        dest=target.split('#',1)[0]
        if not (p.parent/dest).exists():errors.append(f'{p.relative_to(ROOT)}: missing {dest}')
    for match in re.finditer(r'```python\n(.*?)\n```',text,re.S):
        try:ast.parse(match.group(1));blocks+=1
        except SyntaxError as e:errors.append(f'{p.relative_to(ROOT)}: invalid Python: {e}')
for p in sorted((ROOT/'examples').glob('*.ipynb')):
    n=json.loads(p.read_text());assert n['nbformat']==4
    for cell in n['cells']:
        if cell['cell_type']=='code':
            try:ast.parse(''.join(cell['source']));blocks+=1
            except SyntaxError as e:errors.append(f'{p.name}: invalid cell: {e}')
if errors:raise SystemExit('\n'.join(errors))
print(f'PASS: local document links and {blocks} Python/Notebook code blocks')
