import shutil
import os
from pathlib import Path

ckpt_dir = Path(r'configs/best_c_index.pth')
output_file = Path(r'checkpoints/gliosurv_best.pth')

# 创建 checkpoints 目录
output_file.parent.mkdir(parents=True, exist_ok=True)

print(f'Repackaging checkpoint from: {ckpt_dir}')
print(f'Output file: {output_file}')

try:
    # 使用 shutil.make_archive 重新打包
    # 注意：make_archive 会自动添加 .zip 扩展，所以我们先去掉 .pth
    base_name = str(output_file.with_suffix(''))
    print(f'Creating zip archive...')
    
    # 删除旧文件
    if output_file.exists():
        os.remove(output_file)
    
    # 打包
    archive_path = shutil.make_archive(base_name, 'zip', str(ckpt_dir.parent), ckpt_dir.name)
    print(f'Created: {archive_path}')
    
    # 重命名为 .pth
    os.rename(archive_path, output_file)
    print(f'✓ Renamed to: {output_file}')
    
    # 验证
    print(f'\n验证新文件：')
    print(f'File exists: {output_file.exists()}')
    print(f'File size: {output_file.stat().st_size / (1024*1024):.2f} MB')
    
    print(f'\n✓ 成功！')
    print(f'\n下一步：')
    print(f'1. 在 configs/gliosurv.yaml 中设置：')
    print(f'   pretrain: checkpoints/gliosurv_best.pth')
    print(f'   或')
    print(f'   pretrain: ./checkpoints/gliosurv_best.pth')
    
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
