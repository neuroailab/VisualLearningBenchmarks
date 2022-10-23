from openselfsup.datasets.data_sources.imagenet import ImageNetBatchLD
import pdb


def batch_ld_test():
    data_train_list = 'data/imagenet/meta/train.txt'
    data_train_root = 'data/imagenet/train'
    saycam_data = ImageNetBatchLD(
            root=data_train_root, list_file=data_train_list, batch_size=256,
            no_cate_per_batch=10, 
            memcached=False,
            mclient_path='/mnt/lustre/share/memcached_client')
    return saycam_data


def main():
    imgnt_data = batch_ld_test()
    pdb.set_trace()
    pass


if __name__ == '__main__':
    main()
