#!/usr/bin/env python3
"""
Non-destructive repo cleanup utility.

Default is dry-run: prints what would be archived. Use --apply to move files
into timestamped folders under ./archive/.

Categories
- results/: resynth_results_*
- outputs/: outputs*, outputs_* (models, images, audio)
- debug_scripts/: debug_*.py (kept by default unless --include-debug)
- variants/: multiple enhanced_* augmentation script variants
- misc/: common noise files (.DS_Store)

Examples
  python tools/repo_cleanup.py           # dry run
  python tools/repo_cleanup.py --apply   # archive candidates
  python tools/repo_cleanup.py --include-debug --apply
"""

import argparse
import os
import shutil
from pathlib import Path
from datetime import datetime


def find_candidates(include_debug: bool):
    root = Path('.')
    # Results dirs
    results = [p for p in root.glob('resynth_results_*') if p.is_dir()]
    # Output dirs (keep top-level 'outputs' if you actively use it)
    outputs = [p for p in root.iterdir() if p.is_dir() and p.name.startswith('outputs')]
    # Variant augmentation scripts
    variants = [
        Path('enhanced_polymax_augmentation.py'),
        Path('enhanced_polymax_augmentation_fixed.py'),
        Path('enhanced_polymax_augmentation_corrected.py'),
        Path('tools/polymax_augment_isolated_fixed.py'),
    ]
    variants = [p for p in variants if p.exists()]
    # Debug scripts
    debug = [p for p in root.glob('debug_*.py') if p.is_file()]
    # Misc noise
    misc = [p for p in root.glob('**/.DS_Store') if p.is_file()]
    # Large top-level test audio (optional)
    audio = [Path("11 Missing Project - Poisson D'Avril (Galaxy dub) - Crop #1.wav")]
    audio = [p for p in audio if p.exists()]
    # Obsolete dataset directories
    obsolete_datasets = [
        Path('datasets/polymax_aug_small'),
        Path('datasets/polymax_augmented'),
        Path('datasets/polymax_iso_small'),
        Path('datasets/polymax_preset_jitter'),
        Path('datasets/polymax_sweeps'),
    ]
    obsolete_datasets = [p for p in obsolete_datasets if p.exists()]

    return {
        'results': results,
        'outputs': outputs,
        'variants': variants,
        'debug_scripts': debug if include_debug else [],
        'misc': misc,
        'samples': audio,
        'obsolete_datasets': obsolete_datasets,
    }


def archive(paths, archive_root: Path, subdir: str):
    dest_dir = archive_root / subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        try:
            target = dest_dir / p.name
            # Avoid overwriting; add numeric suffix if needed
            i = 1
            t = target
            while t.exists():
                t = dest_dir / f"{p.stem}_{i}{p.suffix}"
                i += 1
            shutil.move(str(p), str(t))
            print(f"archived: {p} -> {t}")
        except Exception as e:
            print(f"WARN: could not archive {p}: {e}")


def main():
    ap = argparse.ArgumentParser(description='Repo cleanup (non-destructive archive)')
    ap.add_argument('--apply', action='store_true', help='Move candidates to ./archive/<date>/')
    ap.add_argument('--include-debug', action='store_true', help='Include debug_*.py in archive')
    ap.add_argument('--archive-root', default='archive', help='Archive base directory')
    args = ap.parse_args()

    c = find_candidates(include_debug=args.include_debug)
    print("Cleanup candidates:")
    for k, lst in c.items():
        print(f"- {k}: {len(lst)} items")
        for p in lst[:8]:
            print(f"    {p}")
        if len(lst) > 8:
            print(f"    ... and {len(lst) - 8} more")

    if not args.apply:
        print("\nDry run. Use --apply to archive candidates.")
        return

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    archive_root = Path(args.archive_root) / stamp
    for key in ['results', 'outputs', 'variants', 'debug_scripts', 'misc', 'samples', 'obsolete_datasets']:
        archive(c.get(key, []), archive_root, key)
    print(f"\nArchive complete: {archive_root}")


if __name__ == '__main__':
    main()
