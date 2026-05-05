import os
import shutil
import pandas as pd
import numpy as np
import nibabel as nib
from pathlib import Path

"""
Prepare test dataset for a single-modal model (T1CE only) with multi-class tumor mask.

Actions:
1. Read `data/clinical_data.xlsx` and create `data/test_label.csv` with only required columns.
2. Reorganize `data/T1C` and `data/Mask` into `data/test/preprocessed/<patient>/` containing:
   t1ce.nii.gz, brain_mask.nii.gz (binary: 1 for any tumor region), tumor_mask.nii.gz (original labels 0/1/2/3)

Usage:
    python scripts/prepare_test_data.py --excel data/clinical_data.xlsx --dataroot data
"""

def prepare(excel_path: str = 'data/clinical_data.xlsx', dataroot: str = 'data'):
    excel_path = Path(excel_path)
    dataroot = Path(dataroot)

    if not excel_path.exists():
        raise FileNotFoundError(f"Excel not found: {excel_path}")

    df = pd.read_excel(excel_path)

    # Accept either OS/OS.time or death_event/death_duration_month (case-insensitive)
    col_map = {c.lower(): c for c in df.columns}
    if 'patient_id' not in col_map:
        raise ValueError('Excel must contain column: patient_id')

    pid_col = col_map['patient_id']
    age_col = col_map.get('age', None)
    sex_col = col_map.get('sex', None)
    os_col = col_map.get('os', None)
    ost_col = col_map.get('os.time', None)
    death_event_col = col_map.get('death_event', None)
    death_duration_col = col_map.get('death_duration_month', None)

    if (os_col is None or ost_col is None) and (death_event_col is None or death_duration_col is None):
        raise ValueError('Excel must contain either columns: OS + OS.time, or death_event + death_duration_month')

    # Build label frame with only required fields
    records = []
    for _, row in df.iterrows():
        raw_pid = row[pid_col]
        if pd.isna(raw_pid):
            continue
        if isinstance(raw_pid, (int, float)):
            pid = f"C{int(raw_pid)}"
        else:
            pid_str = str(raw_pid).strip()
            pid = pid_str if pid_str.upper().startswith('C') else f"C{pid_str}"

        # Get age and sex if available
        age = int(row[age_col]) if age_col is not None and not pd.isna(row[age_col]) else -1
        sex = int(row[sex_col]) if sex_col is not None and not pd.isna(row[sex_col]) else -1

        if os_col is not None and ost_col is not None:
            death_event = int(row[os_col]) if not pd.isna(row[os_col]) else -1
            death_duration_month = int(row[ost_col]) if not pd.isna(row[ost_col]) else -1
        else:
            death_event = int(row[death_event_col]) if not pd.isna(row[death_event_col]) else -1
            death_duration_month = int(row[death_duration_col]) if not pd.isna(row[death_duration_col]) else -1

        rec = {
            'patient_id': pid,
            'age': age,
            'sex': sex,
            'death_event': death_event,
            'death_duration_month': death_duration_month,
        }
        records.append(rec)

    out_df = pd.DataFrame.from_records(records)
    out_csv = dataroot / 'test_label.csv'
    out_df.to_csv(out_csv, index=False)
    print(f'Wrote labels to {out_csv}')
    print(f'Columns: {list(out_df.columns)}')

    # Reorganize images
    t1c_dir = dataroot / 'T1C'
    mask_dir = dataroot / 'Mask'
    out_preproc = dataroot / 'test' / 'preprocessed'
    out_preproc.mkdir(parents=True, exist_ok=True)

    # Find available patient ids from existing T1C files
    available = []
    for f in t1c_dir.glob('*.nii*'):
        name = f.name
        if name.endswith('.nii.gz'):
            pid = name[:-7]
        elif name.endswith('.nii'):
            pid = name[:-4]
        else:
            continue
        available.append(pid)

    for pid in available:
        src_img = t1c_dir / f'{pid}.nii.gz'
        if not src_img.exists():
            src_img = t1c_dir / f'{pid}.nii'
        # Look for mask with several possible patterns
        src_mask = mask_dir / f'{pid}.nii.gz'
        if not src_mask.exists():
            src_mask = mask_dir / f'{pid}.nii'
        if not src_mask.exists():
            src_mask = mask_dir / f'val_{pid}.nii.gz'
        if not src_mask.exists():
            src_mask = mask_dir / f'val_{pid}.nii'

        if not src_img.exists():
            print(f'Warning: missing image for {pid}, skipping')
            continue

        patient_out = out_preproc / pid
        patient_out.mkdir(parents=True, exist_ok=True)

        # Copy T1CE only (no duplication to other modalities)
        shutil.copy2(src_img, patient_out / 't1ce.nii.gz')

        # Process mask
        if src_mask.exists():
            # Load mask with nibabel
            try:
                mask_nii = nib.load(src_mask)
                mask_data = mask_nii.get_fdata().astype(np.uint8)

                # tumor_mask: keep original multi-class labels (0,1,2,3)
                tumor_mask_nii = nib.Nifti1Image(mask_data, mask_nii.affine, mask_nii.header)
                nib.save(tumor_mask_nii, patient_out / 'tumor_mask.nii.gz')

                # brain_mask: binary mask (1 for any tumor region, i.e. mask > 0)
                brain_data = (mask_data > 0).astype(np.uint8)
                brain_mask_nii = nib.Nifti1Image(brain_data, mask_nii.affine, mask_nii.header)
                nib.save(brain_mask_nii, patient_out / 'brain_mask.nii.gz')

                print(f'Prepared patient {pid} (T1CE + multi-class mask)')
            except Exception as e:
                print(f'Error processing mask for {pid}: {e}')
                # Fallback: copy raw mask as both (but won't have binary brain mask)
                shutil.copy2(src_mask, patient_out / 'tumor_mask.nii.gz')
                shutil.copy2(src_mask, patient_out / 'brain_mask.nii.gz')
                print(f'  Used fallback copy for {pid}')
        else:
            print(f'Warning: missing mask for {pid}, using dummy masks (image placeholder)')
            # Create dummy masks (copy T1CE as mask placeholder - not ideal)
            shutil.copy2(src_img, patient_out / 'brain_mask.nii.gz')
            shutil.copy2(src_img, patient_out / 'tumor_mask.nii.gz')

    print('Done.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--excel', type=str, default='data/clinical_data.xlsx')
    parser.add_argument('--dataroot', type=str, default='data')
    args = parser.parse_args()
    prepare(args.excel, args.dataroot)
