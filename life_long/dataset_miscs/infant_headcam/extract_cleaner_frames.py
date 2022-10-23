import numpy as np
import math
import os
import sys
import argparse
import pdb
from tqdm import tqdm
from multiprocessing import Pool
import functools

NUM_THREADS = 20
USER_NAME = os.getlogin()


def get_parser():
    parser = argparse.ArgumentParser(
            description='The script to extract the jpgs from videos')
    parser.add_argument(
            '--raw_mp4_dir',
            default='/mnt/fs1/Dataset/infant_headcam/infant_headcam/samcam',
            type=str, action='store',
            help='Directory to hold the downloaded videos')
    parser.add_argument(
            '--jpg_dir',
            default=f'/data5/{USER_NAME}/Dataset/infant_headcam/jpgs_extracted/Samcam',
            type=str, action='store',
            help='Directory to hold the extracted jpgs, rescaled')
    parser.add_argument(
            '--sta_idx', default=0, type=int, action='store',
            help='Start index for downloading')
    parser.add_argument(
            '--len_idx', default=5, type=int,
            action='store', help='Length of index of downloading')
    parser.add_argument(
            '--check', default=0, type=int, action='store',
            help='Whether checking the existence')
    return parser


def extract_one_video(curr_indx, args, video_list):
    mp4_name = video_list[curr_indx]
    save_folder = os.path.join(
            args.jpg_dir,
            mp4_name[:-4])
    mp4_path = os.path.join(args.raw_mp4_dir, mp4_name)

    if os.path.exists(save_folder) and args.check==1:
        return

    os.system('mkdir -p %s' % save_folder)

    tmpl = '%06d.jpg'
    cmd = 'ffmpeg -i {} -vf scale=-1:320,fps=25 {} > /dev/null 2>&1'.format(
            mp4_path,
            os.path.join(save_folder, tmpl))
    os.system(cmd)


def main():
    parser = get_parser()
    args = parser.parse_args()

    video_list = os.listdir(args.raw_mp4_dir)
    video_list = sorted(video_list)
    curr_len = min(len(video_list) - args.sta_idx, args.len_idx)

    _func = functools.partial(
            extract_one_video, args=args, 
            video_list=video_list)
    p = Pool(NUM_THREADS)
    r = list(tqdm(
        p.imap(
            _func,
            range(args.sta_idx, args.sta_idx + curr_len)),
        total=curr_len))


if __name__ == '__main__':
    main()
