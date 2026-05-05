#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
完整预处理脚本：
1. 重采样到目标 spacing (1.0, 1.0, 4.0) mm
2. 基于 brain mask 裁剪 ROI（使用 bounding box，安全提取）
3. 中心裁剪/填充到目标尺寸 (160, 192, 40)
4. 在 ROI 内 Z-score 归一化（仅 t1ce）
5. 保持 tumor_mask 多类别标签（最近邻插值）
输入目录结构：每个患者子文件夹包含 t1ce.nii.gz, tumor_mask.nii.gz, brain_mask.nii.gz
输出：处理后的相同文件名
"""

import os
import numpy as np
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool

# 目标参数（与 vit3d.yaml 一致）
TARGET_SPACING = (1.0, 1.0, 4.0)          # (x, y, z)
TARGET_SHAPE   = (160, 192, 40)           # (x, y, z)

def resample_to_spacing(image_sitk, new_spacing, interpolator=sitk.sitkLinear):
    """重采样图像到新的体素间距"""
    orig_spacing = image_sitk.GetSpacing()
    orig_size = image_sitk.GetSize()
    new_size = [int(round(orig_size[i] * orig_spacing[i] / new_spacing[i])) for i in range(3)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(image_sitk.GetOrigin())
    resampler.SetOutputDirection(image_sitk.GetDirection())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image_sitk)

def get_roi_from_mask(mask_sitk, label_value=1):
    """从 mask 中提取包含指定标签的 bounding box 的索引和尺寸"""
    # 确保 mask 是整数类型
    if mask_sitk.GetPixelID() != sitk.sitkUInt8:
        mask_sitk = sitk.Cast(mask_sitk, sitk.sitkUInt8)
    # 统计标签
    label_stats = sitk.LabelShapeStatisticsImageFilter()
    label_stats.Execute(mask_sitk)
    if not label_stats.HasLabel(label_value):
        # 如果没有该标签，返回全图范围
        size = mask_sitk.GetSize()
        return (0, 0, 0), size
    bbox = label_stats.GetBoundingBox(label_value)  # (x, y, z, width, height, depth)
    # bbox: (min_x, min_y, min_z, size_x, size_y, size_z)
    min_idx = (bbox[0], bbox[1], bbox[2])
    size = (bbox[3], bbox[4], bbox[5])
    return min_idx, size

def extract_roi(image_sitk, min_idx, size):
    """根据索引和尺寸提取 ROI（不越界）。
    使用基于 numpy 的裁剪实现以避免 SimpleITK RegionOfInterest 的类型问题。
    注意 SimpleITK 的 GetArrayFromImage 返回数组顺序为 (z, y, x)。
    """
    # 图像尺寸和输入索引均为 (x, y, z)
    img_size = image_sitk.GetSize()
    start_x = max(0, int(min_idx[0]))
    start_y = max(0, int(min_idx[1]))
    start_z = max(0, int(min_idx[2]))
    size_x = int(size[0])
    size_y = int(size[1])
    size_z = int(size[2])

    end_x = min(img_size[0], start_x + size_x)
    end_y = min(img_size[1], start_y + size_y)
    end_z = min(img_size[2], start_z + size_z)

    # 转为 numpy 并按 (z,y,x) 裁剪
    arr = sitk.GetArrayFromImage(image_sitk)
    z0, z1 = start_z, end_z
    y0, y1 = start_y, end_y
    x0, x1 = start_x, end_x
    cropped = arr[z0:z1, y0:y1, x0:x1]

    cropped_img = sitk.GetImageFromArray(cropped)
    # 保持原图的空间信息，但需要调整 origin：原始 origin + start * spacing
    spacing = image_sitk.GetSpacing()
    origin = list(image_sitk.GetOrigin())
    origin[0] = origin[0] + start_x * spacing[0]
    origin[1] = origin[1] + start_y * spacing[1]
    origin[2] = origin[2] + start_z * spacing[2]
    cropped_img.SetSpacing(spacing)
    cropped_img.SetOrigin(tuple(origin))
    cropped_img.SetDirection(image_sitk.GetDirection())
    return cropped_img

def center_crop_or_pad(image_sitk, target_size):
    """中心裁剪或零填充图像到目标尺寸（非递归、鲁棒实现）。
    优先使用 SimpleITK 的 Crop/Pad；若仍有偏差，使用 numpy 精确中心化复制。
    """
    curr_size = list(image_sitk.GetSize())
    if tuple(curr_size) == tuple(target_size):
        return image_sitk

    # 计算裁剪和填充量（基于当前尺寸 -> target）
    lower_crop = [0, 0, 0]
    upper_crop = [0, 0, 0]
    lower_pad = [0, 0, 0]
    upper_pad = [0, 0, 0]

    for i in range(3):
        diff = curr_size[i] - target_size[i]
        if diff > 0:
            # 需要裁剪: 去掉两侧各 diff//2 和 diff-diff//2
            start = diff // 2
            end = diff - start
            lower_crop[i] = int(start)
            upper_crop[i] = int(end)
        elif diff < 0:
            pad = -diff
            lower_pad[i] = int(pad // 2)
            upper_pad[i] = int(pad - lower_pad[i])

    if any(lower_crop) or any(upper_crop):
        crop = sitk.CropImageFilter()
        crop.SetLowerBoundaryCropSize([int(x) for x in lower_crop])
        crop.SetUpperBoundaryCropSize([int(x) for x in upper_crop])
        image_sitk = crop.Execute(image_sitk)

    if any(lower_pad) or any(upper_pad):
        padf = sitk.ConstantPadImageFilter()
        padf.SetPadLowerBound([int(x) for x in lower_pad])
        padf.SetPadUpperBound([int(x) for x in upper_pad])
        padf.SetConstant(0)
        image_sitk = padf.Execute(image_sitk)

    # 最后检查尺寸，若仍不匹配则用 numpy 精确中心化复制
    if tuple(image_sitk.GetSize()) != tuple(target_size):
        arr = sitk.GetArrayFromImage(image_sitk)  # z,y,x
        tz, ty, tx = target_size[2], target_size[1], target_size[0]
        # 注意坐标顺序转换
        target_arr = np.zeros((tz, ty, tx), dtype=arr.dtype)

        cz = arr.shape[0]
        cy = arr.shape[1]
        cx = arr.shape[2]

        start_z = max(0, (tz - cz) // 2) if tz >= cz else 0
        start_y = max(0, (ty - cy) // 2) if ty >= cy else 0
        start_x = max(0, (tx - cx) // 2) if tx >= cx else 0

        dst_z0 = start_z
        dst_y0 = start_y
        dst_x0 = start_x
        src_z0 = 0 if tz >= cz else (cz - tz) // 2
        src_y0 = 0 if ty >= cy else (cy - ty) // 2
        src_x0 = 0 if tx >= cx else (cx - tx) // 2

        copy_z = min(tz, cz)
        copy_y = min(ty, cy)
        copy_x = min(tx, cx)

        target_arr[dst_z0:dst_z0+copy_z, dst_y0:dst_y0+copy_y, dst_x0:dst_x0+copy_x] = \
            arr[src_z0:src_z0+copy_z, src_y0:src_y0+copy_y, src_x0:src_x0+copy_x]

        new_img = sitk.GetImageFromArray(target_arr)
        new_img.SetSpacing(image_sitk.GetSpacing())
        new_img.SetOrigin(image_sitk.GetOrigin())
        new_img.SetDirection(image_sitk.GetDirection())
        return new_img

    return image_sitk

def normalize_intensity(image_sitk, mask_sitk):
    """在 mask 区域内进行 Z-score 归一化，背景设为 0"""
    # 如果 mask 与 image 的空间/尺寸不一致，先把 mask 重采样到 image 的空间
    if mask_sitk.GetSize() != image_sitk.GetSize() or mask_sitk.GetSpacing() != image_sitk.GetSpacing() or mask_sitk.GetOrigin() != image_sitk.GetOrigin():
        mask_sitk = sitk.Resample(mask_sitk, image_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask_sitk.GetPixelID())

    img_arr = sitk.GetArrayFromImage(image_sitk)
    mask_arr = sitk.GetArrayFromImage(mask_sitk)
    roi_pixels = img_arr[mask_arr > 0]
    if roi_pixels.size == 0:
        return image_sitk
    mean = roi_pixels.mean()
    std = roi_pixels.std()
    if std < 1e-6:
        return image_sitk
    img_arr = (img_arr - mean) / std
    img_arr[mask_arr <= 0] = 0
    normalized = sitk.GetImageFromArray(img_arr)
    normalized.CopyInformation(image_sitk)
    return normalized

def preprocess_patient(patient_dir, target_root):
    patient_id = os.path.basename(patient_dir)
    target_dir = os.path.join(target_root, patient_id)
    os.makedirs(target_dir, exist_ok=True)

    t1ce_path = os.path.join(patient_dir, 't1ce.nii.gz')
    mask_path = os.path.join(patient_dir, 'tumor_mask.nii.gz')
    brain_path = os.path.join(patient_dir, 'brain_mask.nii.gz')

    for p in [t1ce_path, mask_path, brain_path]:
        if not os.path.exists(p):
            print(f"Missing {p} for {patient_id}, skipping")
            return

    # 读取图像
    t1ce = sitk.ReadImage(t1ce_path)
    mask = sitk.ReadImage(mask_path)
    brain = sitk.ReadImage(brain_path)

    # 确保 mask 为整数类型（用于重采样和 ROI 提取）
    if mask.GetPixelID() != sitk.sitkUInt8:
        mask = sitk.Cast(mask, sitk.sitkUInt8)
    if brain.GetPixelID() != sitk.sitkUInt8:
        brain = sitk.Cast(brain, sitk.sitkUInt8)

    # 1. 重采样到目标 spacing
    t1ce_res = resample_to_spacing(t1ce, TARGET_SPACING, sitk.sitkLinear)
    mask_res = resample_to_spacing(mask, TARGET_SPACING, sitk.sitkNearestNeighbor)
    brain_res = resample_to_spacing(brain, TARGET_SPACING, sitk.sitkNearestNeighbor)

    # 2. 基于 brain mask 裁剪 ROI
    min_idx, roi_size = get_roi_from_mask(brain_res, label_value=1)
    t1ce_roi = extract_roi(t1ce_res, min_idx, roi_size)
    mask_roi = extract_roi(mask_res, min_idx, roi_size)
    brain_roi = extract_roi(brain_res, min_idx, roi_size)

    # 3. 强度归一化（在 brain_roi 范围内）
    t1ce_norm = normalize_intensity(t1ce_roi, brain_roi)

    # 4. 统一尺寸
    t1ce_final = center_crop_or_pad(t1ce_norm, TARGET_SHAPE)
    mask_final = center_crop_or_pad(mask_roi, TARGET_SHAPE)
    brain_final = center_crop_or_pad(brain_roi, TARGET_SHAPE)

    # 5. 保存结果
    sitk.WriteImage(t1ce_final, os.path.join(target_dir, 't1ce.nii.gz'))
    sitk.WriteImage(mask_final, os.path.join(target_dir, 'tumor_mask.nii.gz'))
    sitk.WriteImage(brain_final, os.path.join(target_dir, 'brain_mask.nii.gz'))

    print(f"Processed {patient_id}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Complete preprocessing for T1CE + masks')
    parser.add_argument('--source_root', required=True, help='Source directory with patient subfolders')
    parser.add_argument('--target_root', required=True, help='Output directory')
    parser.add_argument('--processes', type=int, default=4, help='Number of parallel processes')
    args = parser.parse_args()

    os.makedirs(args.target_root, exist_ok=True)
    patient_dirs = [d for d in glob(os.path.join(args.source_root, '*')) if os.path.isdir(d)]
    print(f"Found {len(patient_dirs)} patients")

    if args.processes == 1:
        for d in tqdm(patient_dirs):
            preprocess_patient(d, args.target_root)
    else:
        with Pool(processes=args.processes) as pool:
            tasks = [(d, args.target_root) for d in patient_dirs]
            for _ in tqdm(pool.starmap(preprocess_patient, tasks), total=len(patient_dirs)):
                pass

    print("Done.")