#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, sys
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path('.')
out={}
for p in sorted(root.rglob('*')):
    if p.is_file() and '.git' not in p.parts:
        out[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
print(json.dumps(out,indent=2,sort_keys=True))
