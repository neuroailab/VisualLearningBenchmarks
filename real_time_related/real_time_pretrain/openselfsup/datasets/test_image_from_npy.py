import openselfsup.datasets.image_from_npy as image_from_npy
import pdb


def main():
    data_path = 'data/imagenet/in_val_processed.npy'
    meta_path = 'data/imagenet/meta/part_train_val_labeled.txt'
    img_norm_cfg = dict(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    pipeline = [
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
    ]
    dataset = image_from_npy.ImageFromNpy(
            data_path, pipeline, meta_path)


if __name__ == '__main__':
    main()
