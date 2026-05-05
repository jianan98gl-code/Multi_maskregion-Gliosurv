import nibabel as nib
import numpy as np

# 加载你的mask
mask_img = nib.load('data/Mask/val_C1.nii.gz')
mask_data = mask_img.get_fdata()

# 查看有哪些标签值
unique_labels = np.unique(mask_data)
print(f"Mask中的标签值: {unique_labels}")

# 统计每个标签的体素数
for label in unique_labels:
    count = np.sum(mask_data == label)
    print(f"Label {label}: {count} 体素")