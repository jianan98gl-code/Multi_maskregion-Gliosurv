import argparse
import os
from types import SimpleNamespace


def build_args(dataroot: str):
    return SimpleNamespace(dataroot=dataroot, distributed=False)


def summarize_sample(tag: str, sample: dict) -> None:
    print(f"\n[{tag}]")
    print(f"patient_id: {sample.get('patient_id')}")
    print(f"death_duration_month: {sample.get('death_duration_month')}")
    print(f"death_event: {sample.get('death_event')}")

    input_tensor = sample.get('input')
    if input_tensor is None:
        print('input: missing')
        return

    print(f"batch input shape: {tuple(input_tensor.shape)}")
    print(f"input dtype: {input_tensor.dtype}")

    sample_tensor = input_tensor[0] if input_tensor.ndim == 5 else input_tensor
    print(f"single-sample shape: {tuple(sample_tensor.shape)}")
    print(f"single-sample min/max: {float(sample_tensor.min())} / {float(sample_tensor.max())}")

    if sample_tensor.shape[0] >= 4:
        summaries = []
        for index, name in enumerate(['t1ce', 'mask_necrosis', 'mask_edema', 'mask_enhancing']):
            channel = sample_tensor[index]
            summaries.append(
                f"{name}: min={float(channel.min())}, max={float(channel.max())}, unique={sorted(channel.unique().tolist())[:10]}"
            )
        print(' | '.join(summaries))


def main():
    parser = argparse.ArgumentParser(description='Quickly verify MRI dataloader and transforms.')
    parser.add_argument('--dataroot', type=str, default='data', help='Root directory containing train/valid folders and label CSV files')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for the check loader')
    parser.add_argument('--workers', type=int, default=0, help='Number of dataloader workers')
    args = parser.parse_args()

    loader_workers = max(1, args.workers)

    if not os.path.isdir(args.dataroot):
        raise FileNotFoundError(f'Dataroot does not exist: {args.dataroot}')

    from src.data.mri_datasets import get_train_loader, get_val_loader
    from src.data.mri_transforms import get_classification_train_transforms, get_val_transforms

    loader_args = build_args(args.dataroot)

    train_loader = get_train_loader(
        loader_args,
        batch_size=args.batch_size,
        workers=loader_workers,
        train_transform=get_classification_train_transforms(),
    )
    val_loader = get_val_loader(
        loader_args,
        batch_size=args.batch_size,
        workers=loader_workers,
        val_transform=get_val_transforms(),
    )

    train_sample = next(iter(train_loader))
    val_sample = next(iter(val_loader))

    summarize_sample('train', train_sample)
    summarize_sample('valid', val_sample)

    print('\nData loading check finished successfully.')


if __name__ == '__main__':
    main()