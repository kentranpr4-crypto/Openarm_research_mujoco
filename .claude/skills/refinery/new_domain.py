#!/usr/bin/env python3
"""Create a domain outside live data; refuse existing targets, traversal and option-like names."""
import argparse
import re
from pathlib import Path
import yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('name', help='lowercase letters, digits and hyphens')
    ap.add_argument('label', nargs='?')
    ap.add_argument('--output-root', type=Path, default=Path(__file__).resolve().parent / 'domains')
    args = ap.parse_args()
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,63}', args.name):
        ap.error('name must use lowercase letters, digits and hyphens, without path separators')
    target = args.output_root / args.name
    if target.exists():
        ap.error(f'target already exists: {target}')
    target.mkdir(parents=True)
    (target / 'snapshots').mkdir()
    cfg = {'domain': args.name, 'entity_label': args.label or args.name,
           'schema': {'type_field': None, 'fields': ['name', 'category', 'cost']},
           'rollup_field': None, 'universe': {'estimate': None, 'basis': 'Chưa xác định; không công bố coverage.'},
           'refresh_days': {'default': 180}, 'alias_map': {}, 'ambiguous_clusters': []}
    (target / 'domain.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    (target / 'claims.jsonl').write_text('', encoding='utf-8')
    (target / 'README.md').write_text('# Domain mới\n\nĐiền schema và nguồn; claims.jsonl ban đầu rỗng, không có claim giả.\n'
        'Mẫu số chỉ điền khi có basis; refresh_days là metadata, chưa có gate độ tươi.\n'
        'Chạy refinery.py rồi bites.py với đường dẫn domain này. Đọc README của hộp Refinery.\n', encoding='utf-8')
    print(f'Created {target}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
