from openselfsup.datasets.saycam_vectors import *
import pdb
root = '/mnt/fs1/Dataset/infant_headcam/embeddings/'


def embd_test():
    list_file = '/mnt/fs1/Dataset/infant_headcam/embd_train_meta.txt'
    #list_file = '/mnt/fs1/Dataset/infant_headcam/embd_val_meta.txt'
    #saycam_data = SAYCamSeqVecDataset(
    #        root=root, list_file=list_file, 
    #        which_model='simclr_sy', seq_len=32)
    saycam_data = OSFilterVLenSCSeqVec(
            root=root, list_file=list_file, 
            which_model='simclr_mst_pair_ft_in', 
            seq_len=64, min_seq_len=32, batch_size=1024,
            cache_folder='/mnt/fs1/Dataset/infant_headcam/embeddings_related/simclr_mst_pair_ft_in/seq_32_64_bs_1024',
            )
    return saycam_data


def main():
    saycam_data = embd_test()
    pdb.set_trace()
    pass


def run_set_epochs(max_epochs=400):
    saycam_data = embd_test()
    for _e in range(max_epochs):
        print('At epoch', _e)
        saycam_data.set_epoch(_e)


if __name__ == '__main__':
    #main()
    run_set_epochs()
