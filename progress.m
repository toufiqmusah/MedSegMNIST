%% ============================================================================
%% MedSegMNIST — Agent Handover Document ("Second Brain")
%% ============================================================================
% This file is the canonical handover for any new agent. It captures all
% decisions, architecture, gotchas, and next steps so work can resume
% without context loss.
%
% Last updated: 2026-06-18
% Project: /teamspace/studios/this_studio/MedSegMNIST/
%% ============================================================================

fprintf('\n');
fprintf('╔══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║           MedSegMNIST — Agent Handover / Second Brain              ║\n');
fprintf('╚══════════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 1.  DATASET STATUS TABLE
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 1. DATASET STATUS ═══════════════════════════════════════════════\n\n');

datasets = {
   1, 'AbdomenSegMNIST3D',  'CT',         '3D', 6, [64, 96, 128, 192, "native"], '✅ registered',        '❌ no data files';
   2, 'BrainSegMNIST3D',    'MRI',        '3D', 4, [96, 128, 224, "native"],      '⬜ 96+128 NPZ exist',   '❌ 224+missing; no native init override';
   3, 'SpineSegMNIST3D',    'MR',         '3D', 3, [64, 96, 128, 192, "native"],  '✅ registered',        '⚠️ FIXME: mask rotated 180°';
   4, 'KneeSegMNIST3D',     'MR',         '3D', 6, [64, 96, 128, 192, "native"],  '✅ registered',        '⚠️ view_axis=1, rot90_k=1';
   5, 'LungSegMNIST2D',     'X-ray',      '2D', 2, [128, 256, 512],               '⬜ 128+256 NPZ exist', '❌ 512 NPZ missing (class says yes)';
   6, 'NucleiSegMNIST2D',   'Pathology',  '2D', 2, [256, 512, "native"],          '✅ registered',        '❌ no data files';
   7, 'PolypSegMNIST2D',    'Endoscopy',  '2D', 2, [128, 256, 512, "native"],     '✅ FULLY LIVE',        '⭐ reference implementation';
   8, 'BreastSegMNIST',     'Ultrasound', '2D', 2, [128, 256, "native"],          '✅ registered',        '❌ no data files; not in datasets/__init__';
   9, 'FundusSegMNIST2D',   'Fundus',     '2D', 2, [256, 512, 1024, "native"],   '✅ registered',        '❌ no data files; not in datasets/__init__';
  10, 'SkinSegMNIST2D',     'Dermoscopy', '2D', 2, [128, 256, 512, "native"],     '✅ registered',        '❌ no data files';
};

for i = 1:size(datasets, 1)
    fprintf('  %2d  %-19s  %-12s  %s  %d classes  →  %s\n', ...
        datasets{i,:});
end

fprintf('\n');
fprintf('  Legend:  ✅ done  ⬜ partial  ❌ missing  ⚠️ known bug\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 2.  WHAT WAS DONE (PolypSegMNIST2D — the only fully live dataset)
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 2. WHAT WAS IMPLEMENTED ════════════════════════════════════════\n\n');

fprintf('  PolypSegMNIST2D (Kvasir-SEG) — fully operational:\n');
fprintf('    • Dataset class at medsegmnist/datasets/endoscopy/polyp.py\n');
fprintf('    • 3 sizes: 128×128, 256×256, 512×512 (NPZ)\n');
fprintf('    • Native resolution: 1000 per-image NPZs in polyp2d_native/\n');
fprintf('    • All-in-one NPZ (train_images + train_masks); split by cv_folds\n');
fprintf('    • 5-fold cross-validation (recently upgraded from 80/20)\n');
fprintf('    • Mask threshold binarization at >128 (JPEG artifact fix)\n');
fprintf('    • JSON metadata with per-sample original_id tracking\n\n');

fprintf('  Visualization overhaul:\n');
fprintf('    • Vibrant 11-color colormap (class 0 = black)\n');
fprintf('    • vmax = max(n_labels - 1, 10) for correct label→color mapping\n');
fprintf('    • Legend/key REMOVED (was causing layout squeeze & color mismatch)\n');
fprintf('    • plot_sample, plot_overlay, plot_grid all use tight_layout\n');
fprintf('    • plot_grid uses n_sub_cols = n_cols*2 (side-by-side) + green borders\n');
fprintf('    • Thread-local viz context auto-detects view_axis / rot90_k\n\n');

fprintf('  Infrastructure:\n');
fprintf('    • notes.md written at /teamspace/studios/this_studio/notes.md\n');
fprintf('    • progress.m handover document (this file)\n');
fprintf('    • Registry auto-generates from @register decorator\n');
fprintf('    • Tests all pass (40/40 pytest, registry, visualize, imports)\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 3.  KEY ARCHITECTURAL DECISIONS
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 3. KEY ARCHITECTURAL DECISIONS ═════════════════════════════════\n\n');

fprintf('  A. NPZ storage convention:\n');
fprintf('     - NPZ files use keys: train_images, train_masks\n');
fprintf('     - NO test_images / test_masks — all data in train_*\n');
fprintf('     - Split is done at load time via cv_folds indices in JSON\n');
fprintf('     - Rationale: consistent with MedSegMNIST base class\n\n');

fprintf('  B. Native resolution pattern:\n');
fprintf('     - Override __init__ to bypass base class NPZ loading\n');
fprintf('     - Per-sample NPZ in {flag}_native/ directory\n');
fprintf('     - Override __getitem__ to load per-sample NPZs\n');
fprintf('     - Set _all_images/_all_masks to np.zeros(n_total) as placeholder\n');
fprintf('     - get_data() raises RuntimeError for native\n');
fprintf('     - Reference: SkinSegMNIST2D (derm.py) and PolypSegMNIST2D\n\n');

fprintf('  C. Cross-validation design:\n');
fprintf('     - Metadata stores fold_0..fold_4 with train[] + test[] index arrays\n');
fprintf('     - Base class _resolve_indices uses fold_0 for split=''train''/''test''\n');
fprintf('     - get_fold(k) returns (Subset, Subset) from any fold\n');
fprintf('     - 5 folds × 200 test samples = full coverage, no overlap\n\n');

fprintf('  D. Colormap design:\n');
fprintf('     - ListedColormap with 11 explicit colors\n');
fprintf('     - vmax = max(n_labels - 1, 10) critical: without this,\n');
fprintf('       n_labels < N causes interpolation that shifts colors\n');
fprintf('     - No legend: archived to fix vertical squeeze + color mismatch\n\n');

fprintf('  E. Data loading flow:\n');
fprintf('     __init__ → _resolve_npz_path() → np.load() → _resolve_indices()\n');
fprintf('     __getitem__ → _indices[index] → _all_images[actual] → normalize\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 4.  GOTCHAS & KNOWN ISSUES
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 4. GOTCHAS & KNOWN ISSUES ══════════════════════════════════════\n\n');

fprintf('  ⚠️  SPINE ORIENTATION: masks rotated 180° at runtime via\n');
fprintf('      np.rot90(mask, k=2, axes=(0,1)). FIXME in spine.py line 89/108.\n');
fprintf('      Until preprocessing is corrected, the FIXME must stay.\n\n');

fprintf('  ⚠️  datasets/__init__.py: exports only 8/10 classes.\n');
fprintf('      BreastSegMNIST and FundusSegMNIST2D are missing — they are\n');
fprintf('      only in the top-level medsegmnist/__init__.py.\n');
fprintf('      FIX: add imports to datasets/__init__.py\n\n');

fprintf('  ⚠️  LungSegMNIST2D size=512: class says available_sizes=[128,256,512]\n');
fprintf('      but no lung2d_512.npz exists. Will FileNotFoundError.\n\n');

fprintf('  ⚠️  BrainSegMNIST3D size=224: class says available but no NPZ.\n');
fprintf('      Also has no __init__ override — native loading not implemented.\n\n');

fprintf('  ⚠️  Abdomen native: uses n_total (not n_samples) in __init__.\n');
fprintf('      Inconsistent with other datasets.\n\n');

fprintf('  ⚠️  All download() methods raise NotImplementedError.\n');
fprintf('      No automatic download pipeline exists.\n\n');

fprintf('  ⚠️  Kvasir-SEG masks are JPEG-compressed with artifacts.\n');
fprintf('      threshold >128 is MANDATORY — raw values include 1-8 and 247-255.\n\n');

fprintf('  ⚠️  Default root is ~/.medsegmnist/ (base.py line 24).\n');
fprintf('      Workspace convention uses root="MedSegMNIST/datasets" or root="datasets".\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 5.  FILE MAP
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 5. FILE MAP ════════════════════════════════════════════════════\n\n');

fprintf('  Code:\n');
fprintf('    medsegmnist/__init__.py          — top-level exports (all 10)\n');
fprintf('    medsegmnist/registry.py          — @register + list_datasets()\n');
fprintf('    medsegmnist/datasets/base.py     — _MedSegMNISTBase, 3D, 2D\n');
fprintf('    medsegmnist/datasets/__init__.py — subpackage exports (only 8!)\n');
fprintf('    medsegmnist/utils/visualize.py   — plot_sample/overlay/grid\n');
fprintf('    medsegmnist/datasets/ct/abdomen.py\n');
fprintf('    medsegmnist/datasets/mri/brain.py\n');
fprintf('    medsegmnist/datasets/mri/spine.py\n');
fprintf('    medsegmnist/datasets/mri/knee.py\n');
fprintf('    medsegmnist/datasets/xray/lung.py\n');
fprintf('    medsegmnist/datasets/pathology/nuclei.py\n');
fprintf('    medsegmnist/datasets/endoscopy/polyp.py    ← most work done here\n');
fprintf('    medsegmnist/datasets/dermoscopy/derm.py\n');
fprintf('    medsegmnist/datasets/ultrasound/breast.py\n');
fprintf('    medsegmnist/datasets/fundus/fives.py\n\n');

fprintf('  Data (live):\n');
fprintf('    MedSegMNIST/datasets/endoscopy/polyp2d_128.npz   (38 MB)\n');
fprintf('    MedSegMNIST/datasets/endoscopy/polyp2d_256.npz   (140 MB)\n');
fprintf('    MedSegMNIST/datasets/endoscopy/polyp2d_512.npz   (491 MB)\n');
fprintf('    MedSegMNIST/datasets/endoscopy/polyp2d_native/   (566 MB, 1000 files)\n');
fprintf('    MedSegMNIST/datasets/endoscopy/polyp2d.json      (metadata + 5-fold CV)\n');
fprintf('    MedSegMNIST/datasets/lung2d_128.npz              (393 MB)\n');
fprintf('    MedSegMNIST/datasets/lung2d_256.npz              (1.5 GB)\n');
fprintf('    MedSegMNIST/datasets/lung2d.json\n');
fprintf('    MedSegMNIST/datasets/brain3d_96.npz              (57 MB)\n');
fprintf('    MedSegMNIST/datasets/brain3d_128.npz             (136 MB)\n');
fprintf('    MedSegMNIST/datasets/brain3d.json\n');
fprintf('    MedSegMNIST/datasets/checksums/                  (SHA-256)\n\n');

fprintf('  Documentation:\n');
fprintf('    /teamspace/studios/this_studio/MedSegMNIST/notes.md    — paper-writing notes\n');
fprintf('    /teamspace/studios/this_studio/MedSegMNIST/progress.m  — this handover\n');
fprintf('    /teamspace/studios/this_studio/MedSegMNIST/tests/       — pytest suite\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 6.  NEXT STEPS / ROADMAP
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 6. PRIORITY NEXT STEPS ════════════════════════════════════════\n\n');

fprintf('  P0 — Get remaining 9 datasets live:\n');
fprintf('    a) Source the original data for each dataset\n');
fprintf('    b) Write preprocessing scripts (model after /tmp/preprocess_kvasir.py)\n');
fprintf('    c) Generate NPZ files at each standardised size\n');
fprintf('    d) Write JSON metadata (model after polyp2d.json)\n');
fprintf('    e) Generate native per-sample NPZs if applicable\n\n');

fprintf('  P1 — Fix the two known dataset __init__ gaps:\n');
fprintf('    a) datasets/__init__.py: add BreastSegMNIST + FundusSegMNIST2D\n');
fprintf('    b) BrainSegMNIST3D: add native __init__ override (like derm.py)\n\n');

fprintf('  P1 — Implement download() methods:\n');
fprintf('    Each dataset needs download logic (Zenodo / HF / direct URL).\n');
fprintf('    Set zenodo_record_id, zenodo_file_ids, hf_repo_id.\n\n');

fprintf('  P2 — Fix LungSegMNIST2D size=512:\n');
fprintf('    Either generate the NPZ or remove 512 from available_sizes.\n\n');

fprintf('  P2 — Fix SpineSegMNIST3D orientation:\n');
fprintf('    Either fix preprocessing to avoid 180° rotation, or make it official.\n\n');

fprintf('  P2 — Fix Abdomen n_total → n_samples inconsistency:\n');
fprintf('    Line 65 of abdomen.py uses n_total instead of n_samples.\n\n');

fprintf('  P3 — Add training examples / model zoo:\n');
fprintf('    Train U-Net on each dataset, report Dice scores, add to examples/\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 7.  CRITICAL CONTEXT FOR NEW AGENT
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 7. CRITICAL CONTEXT ════════════════════════════════════════════\n\n');

fprintf('  Workflow for adding a NEW dataset:\n');
fprintf('    1. Create medsegmnist/datasets/{modality}/{name}.py\n');
fprintf('       with @register, extend MedSegMNIST2D or 3D\n');
fprintf('    2. Add to dataset-specific __init__.py + medsegmnist/datasets/__init__.py\n');
fprintf('    3. Create preprocessing script that generates NPZ files\n');
fprintf('    4. Store NPZs at {root}/{organ}/{flag}_{size}.npz\n');
fprintf('    5. Create {flag}.json with cv_folds + label_names + metadata\n');
fprintf('    6. For native: add __init__ + __getitem__ override (copy derm.py pattern)\n');
fprintf('    7. Add tests and verify with python -m pytest tests/\n\n');

fprintf('  Key import paths:\n');
fprintf('    from medsegmnist import PolypSegMNIST2D          — works (top-level)\n');
fprintf('    from medsegmnist.datasets import PolypSegMNIST2D — works\n');
fprintf('    from medsegmnist.datasets import BreastSegMNIST  — FAILS (missing from __init__)\n\n');

fprintf('  Base class convenions that MUST be followed:\n');
fprintf('    • NPZ keys: always train_images, train_masks (never test_*)\n');
fprintf('    • _resolve_npz_path() returns {flag}_{size}.npz or {flag}_native.npz\n');
fprintf('    • _resolve_indices() uses cv_folds → fold_0 → train/test\n');
fprintf('    • __getitem__ always returns CHW for 2D, CDHW for 3D\n');
fprintf('    • uint8 images are normalized: divide by 255 for RGB, z-score for medical\n');
fprintf('    • Masks are always uint8, never normalized\n');
fprintf('    • All datasets must set: flag, class_name, organ, available_sizes,\n');
fprintf('      n_classes, modality, n_channels\n');
fprintf('    • All datasets must have citation string\n\n');

fprintf('  Testing:\n');
fprintf('    cd /teamspace/studios/this_studio/MedSegMNIST\n');
fprintf('    python -m pytest tests/ -v\n');
fprintf('    Registry tests: tests/test_registry.py\n');
fprintf('    Import tests:   tests/test_imports.py\n');
fprintf('    Viz tests:      tests/test_visualize.py\n');
fprintf('    Dataset tests:  tests/test_datasets.py (auto-generated from registry)\n\n');

fprintf('  Original data source for Kvasir-SEG:\n');
fprintf('    /teamspace/studios/this_studio/kvasir-seg.zip  (46 MB, CC-BY 4.0)\n');
fprintf('    Preprocessing script pattern saved in this document (section 7)\n\n');

%% ────────────────────────────────────────────────────────────────────────────
%% 8.  PREPROCESSING PATTERN (for reference when adding new datasets)
%% ────────────────────────────────────────────────────────────────────────────

fprintf('═══ 8. PREPROCESSING PATTERN REFERENCE ═════════════════════════════\n\n');

fprintf('  Steps that were followed for Kvasir-SEG:\n\n');

fprintf('  1. Extract source zip to /tmp/\n');
fprintf('  2. Shuffle all filenames with fixed seed\n');
fprintf('  3. For each image: read image (RGB), read mask (grayscale)\n');
fprintf('  4. Threshold mask at >128 to binarize\n');
fprintf('  5. Resize bilinear for images, nearest for masks (labels stay intact)\n');
fprintf('  6. Store all samples in train_images / train_masks arrays\n');
fprintf('  7. Save NPZ as np.savez_compressed(path, train_images=..., train_masks=...)\n');
fprintf('  8. Save JSON with n_samples, label_names, cv_folds, per_sample_metadata\n');
fprintf('  9. For native: save each (image, mask) as separate NPZ in {flag}_native/\n\n');

fprintf('  Notes:\n');
fprintf('    • NPZ is uint8 for both images and masks (images normalized on load)\n');
fprintf('    • cv_folds split determines train vs test at load time\n');
fprintf('    • Native NPZs use keys: image, mask (not train_images, train_masks)\n');
fprintf('    • Native NPZs at original float32 pixel values (not uint8 encoded)\n');
fprintf('    • JSON per_sample_metadata tracks original_id for traceability\n\n');

fprintf('╔══════════════════════════════════════════════════════════════════════╗\n');
fprintf('║    End of handover. Any new agent should read this entire file     ║\n');
fprintf('║    before making changes. See notes.md for paper-writing details.   ║\n');
fprintf('╚══════════════════════════════════════════════════════════════════════╝\n');
fprintf('\n');
