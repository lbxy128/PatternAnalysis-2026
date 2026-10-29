"""Build an ISIC 2017 manifest from extracted official archives."""

import argparse
import csv
from pathlib import Path

from audit_labels import parse_labels
from dataset import export
from modules import CLASS_NAMES


def prepare(source: Path, output: Path, evidence: str, include_test=False):
    if not evidence.strip():
        raise ValueError('Record the confirmation for using official splits')
    source = source.resolve()
    if not source.is_dir() or output.resolve().exists():
        raise ValueError('Source must exist and output must be a new directory')
    paths = {}
    for path in source.rglob('*'):
        if path.is_file():
            paths.setdefault(path.name, []).append(path)

    def unique(name):
        matches = paths.get(name, [])
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name}, found {len(matches)}')
        return matches[0]

    rows, label_paths = [], []
    splits = [('train', 'Training'), ('val', 'Validation')]
    if include_test:
        splits.append(('test', 'Test'))
    for split, title in splits:
        labels = unique(f'ISIC-2017_{title}_Part3_GroundTruth.csv')
        parse_labels(labels.read_bytes())
        label_paths.append(labels)
        with labels.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                image_id = row['image_id']
                class_id = 0 if float(row['melanoma']) else 1 if float(row['seborrheic_keratosis']) else 2
                rows.append({'image_id': image_id, 'patient_id': '', 'split': split,
                             'class_id': class_id, 'image_path': str(unique(image_id+'.jpg')),
                             'mask_path': str(unique(image_id+'_segmentation.png'))})
    import tempfile
    import shutil
    with tempfile.TemporaryDirectory() as temp:
        manifest = Path(temp) / 'manifest_official.csv'
        with manifest.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        for labels in label_paths:
            shutil.copy2(labels, Path(temp) / labels.name)
        export(manifest, output, list(CLASS_NAMES), 'official-isic', evidence, temp)
        shutil.copy2(manifest, output / 'manifest_official.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--split-evidence', required=True)
    parser.add_argument('--include-test', action='store_true')
    args = parser.parse_args()
    prepare(args.source, args.output, args.split_evidence, args.include_test)
