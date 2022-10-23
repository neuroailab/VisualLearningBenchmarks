import os
import os.path
from collections import namedtuple
import time
import pdb
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from openselfsup.utils import print_log, build_from_cfg
from torchvision.transforms import Compose
from .registry import DATASETS, VIDEO_PIPELINES

VideoRecord = namedtuple('VideoRecord',
                         ['path', 'num_frames', 'label'])


@DATASETS.register_module
class VideoDataset(data.Dataset):
    '''
    Build pytorch data provider for loading frames from videos

    Args:
        root (str):
            Path to the folder including all jpgs
        metafile (str):
            Path to the metafiles
        frame_interval (int):
            number of frames to skip between two sampled frames, None means
            interval will be computed so that the frames subsampled cover the
            whole video
        frame_start (str):
            Methods of determining which frame to start, RANDOM means randomly
            choosing the starting index, None means the middle of valid range
    '''

    MIN_NUM_FRAMES = 3

    def __init__(
            self, root, metafile, pipeline,
            num_frames=8, frame_interval=None, frame_start=None,
            file_tmpl='{:06d}.jpg', sample_groups=1,
            bin_interval=None,
            trn_style=False, trn_num_frames=8,
            part_vd=None,
            clip_meta=False,
            drop_index=False):

        self.root = root
        self.drop_index = drop_index
        self.metafile = metafile
        self.file_tmpl = file_tmpl

        pipeline = [build_from_cfg(p, VIDEO_PIPELINES) for p in pipeline]
        self.transform = Compose(pipeline)

        self.num_frames = num_frames
        self.frame_interval = frame_interval
        self.frame_start = frame_start
        self.sample_groups = sample_groups
        self.bin_interval = bin_interval
        self.trn_style = trn_style
        self.trn_num_frames = trn_num_frames
        self.part_vd = part_vd
        self.clip_meta = clip_meta
        if 'infant' in self.metafile:
            self.clip_meta = True
        if 'epic_kitchens' in self.metafile:
            self.file_tmpl = 'frame_{:010d}.jpg'
            self.clip_meta = True

        self._parse_list()

    def _load_image(self, directory, idx):
        tmpl = os.path.join(self.root, directory, self.file_tmpl)

        try:
            return Image.open(tmpl.format(idx)).convert('RGB')
        except Exception:
            print('error loading image: {}'.format(tmpl.format(idx)))
            return Image.open(tmpl.format(1)).convert('RGB')

    def __get_interval_valid_range(self, rec_no_frames):
        if self.frame_interval is None:
            interval = rec_no_frames / float(self.num_frames)
        else:
            interval = self.frame_interval
        valid_sta_range = rec_no_frames - (self.num_frames - 1) * interval
        return interval, valid_sta_range

    def _build_bins_for_vds(self):
        self.video_bins = []
        self.bin_curr_idx = []
        self.video_index_offset = []
        curr_index_offset = 0

        for record in self.video_list:
            rec_no_frames = int(record.num_frames)
            _, valid_sta_range = self.__get_interval_valid_range(
                rec_no_frames)
            curr_num_bins = np.ceil(valid_sta_range * 1.0 / self.bin_interval)
            curr_num_bins = int(curr_num_bins)
            curr_bins = [
                (_idx,
                 (self.bin_interval * _idx,
                  min(self.bin_interval * (_idx + 1),
                      valid_sta_range)))
                for _idx in range(curr_num_bins)]

            self.video_bins.append(curr_bins)
            self.bin_curr_idx.append(0)
            self.video_index_offset.append(curr_index_offset)
            np.random.shuffle(self.video_bins[-1])

            curr_index_offset += curr_num_bins
        return curr_index_offset

    def _build_trn_bins(self):
        num_bins = self.trn_num_frames
        half_sec_frames = 12
        all_vds_bin_sta_end = []
        for record in self.video_list:
            rec_no_frames = int(record.num_frames)
            frame_each_bin = min(half_sec_frames, rec_no_frames // num_bins)

            if frame_each_bin == 0:
                all_vds_bin_sta_end.append([])
                continue

            curr_bin_sta_end = []
            for curr_sta in range(0, rec_no_frames, frame_each_bin):
                curr_bin_sta_end.append(
                    (curr_sta,
                     min(curr_sta + frame_each_bin, rec_no_frames)))
            assert len(curr_bin_sta_end) >= num_bins
            all_vds_bin_sta_end.append(curr_bin_sta_end)

        self.all_vds_bin_sta_end = all_vds_bin_sta_end

    def _parse_list(self):
        # check the frame number is >= MIN_NUM_FRAMES
        # usualy it is [video_id, num_frames, class_idx]
        with open(self.metafile) as f:
            len_frame_idx = 1 if not self.clip_meta else 2
            lines = [x.strip().split(' ') for x in f]
            lines = [line for line in lines
                     if int(line[len_frame_idx]) >= self.MIN_NUM_FRAMES]

        record_type = VideoRecord
        if self.clip_meta:
            record_type = ClipRecord
        self.video_list = [record_type(*v) for v in lines]
        if self.part_vd is not None:
            np.random.seed(0)
            now_len = len(self.video_list)
            chosen_indx = sorted(np.random.choice(
                range(now_len), int(now_len * self.part_vd)))
            self.video_list = [self.video_list[_tmp_idx]
                               for _tmp_idx in chosen_indx]

        print('Number of videos: {}'.format(len(self.video_list)))
        if self.bin_interval is not None:
            num_bins = self._build_bins_for_vds()
            print('Number of bins: {}'.format(num_bins))

        if self.trn_style:
            self._build_trn_bins()

    def _get_indices(self, record):
        rec_no_frames = int(record.num_frames)
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        all_offsets = None
        start_interval = valid_sta_range / (1.0 + self.sample_groups)
        for curr_start_group in range(self.sample_groups):
            if self.frame_start is None:
                sta_idx = start_interval * (curr_start_group + 1)
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
            offsets = np.array([int(sta_idx + interval * x)
                                for x in range(self.num_frames)])
            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def _get_binned_indices(self, index, record):
        rec_no_frames = int(record.num_frames)
        interval, valid_sta_range = self.__get_interval_valid_range(
            rec_no_frames)

        _bin_curr_idx = self.bin_curr_idx[index]
        _idx, (_sta_random,
               _end_random) = self.video_bins[index][_bin_curr_idx]
        assert self.frame_start == 'RANDOM', "Binned only supports random!"

        sta_idx = np.random.randint(_sta_random, _end_random)
        offsets = np.array([int(sta_idx + interval * x)
                            for x in range(self.num_frames)])
        self.bin_curr_idx[index] += 1
        if self.bin_curr_idx[index] == len(self.video_bins[index]):
            self.bin_curr_idx[index] = 0
            np.random.shuffle(self.video_bins[index])
        return offsets + 1, _idx + self.video_index_offset[index]

    def _get_valid_video(self, index):
        record = self.video_list[index]
        # check this is a legit video folder
        while not os.path.exists(
                os.path.join(self.root, record.path, self.file_tmpl.format(1))):
            print(
                os.path.join(
                    self.root,
                    record.path,
                    self.file_tmpl.format(1)))
            index = np.random.randint(len(self.video_list))
            record = self.video_list[index]

        if self.frame_interval is not None or self.trn_style:
            needed_frames = self.get_needed_frames()
            while int(record.num_frames) <= needed_frames:
                index = np.random.randint(len(self.video_list))
                record = self.video_list[index]
        return record, index

    def get_needed_frames(self):
        if not self.trn_style:
            needed_frames = self.num_frames * self.frame_interval
        else:
            needed_frames = self.trn_num_frames
        return needed_frames

    def _get_TRN_style_indices(self, index, record):
        curr_bin = self.all_vds_bin_sta_end[index]

        valid_sta_range = len(curr_bin) - self.trn_num_frames + 1
        all_offsets = None
        start_interval = int(valid_sta_range / (1.0 + self.sample_groups))
        for curr_start_group in range(self.sample_groups):
            if self.frame_start is None:
                sta_idx = start_interval * (curr_start_group + 1)
                offsets = np.array([int(np.mean(curr_bin[sta_idx + x]))
                                    for x in range(self.trn_num_frames)])
            elif self.frame_start == 'RANDOM':
                sta_idx = np.random.randint(valid_sta_range)
                offsets = np.array([np.random.randint(*curr_bin[sta_idx + x])
                                    for x in range(self.trn_num_frames)])

            offsets = np.array([np.random.randint(*curr_bin[sta_idx + x])
                                for x in range(self.trn_num_frames)])
            if all_offsets is None:
                all_offsets = offsets
            else:
                all_offsets = np.concatenate([all_offsets, offsets])
        return all_offsets + 1

    def _get_indices_and_instance_index(self, index, record):
        if self.bin_interval is None:
            if not self.trn_style:
                indices = self._get_indices(record)
            else:
                indices = self._get_TRN_style_indices(index, record)
            vd_instance_index = index
        else:
            indices, vd_instance_index = self._get_binned_indices(
                index, record)
        return indices, vd_instance_index

    def __getitem__(self, index):
        record, index = self._get_valid_video(index)

        def _get_frames():
            indices, vd_instance_index = self._get_indices_and_instance_index(
                index, record)

            idx_offset = 0
            if self.clip_meta:
                idx_offset = int(record.start_frame_no)
            frames = self.transform([self._load_image(record.path, int(idx) + idx_offset)
                                     for idx in indices])
            return frames, vd_instance_index
        
        frames_0, _ = _get_frames()
        frames_1, _ = _get_frames()
        if self.num_frames > 1:
            frames_0 = frames_0.unsqueeze(0)
            frames_1 = frames_1.unsqueeze(0)
        frames_cat = torch.cat((frames_0, frames_1), dim=0)
        return dict(img=frames_cat)

    def __len__(self):
        return len(self.video_list)

    def evaluate(self, scores, keyword, logger=None):
        raise NotImplemented
