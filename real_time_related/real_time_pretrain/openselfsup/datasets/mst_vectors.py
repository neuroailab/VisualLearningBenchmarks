import torch
import torch.nn as nn
import os
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from .registry import DATASETS
from torch.utils.data import Dataset
import glob
import csv

@DATASETS.register_module
class MSTSynthVectorDataset(Dataset):
    def __init__(
            self, root, 
            vector_dim=128,
            noise = 0.05,
            SetName = '1',
            data_len=256*5000, 
            lag_set='AllShort_Set2',
            base_dir='LagGenerator',
            bank_size=192,
            real_embds=None,
            ):
        self.bank_size = bank_size
        self.data_len = data_len
        self.root = root
        self.SetName = SetName
        self.lag_set = lag_set
        self.base_dir = base_dir
        
        if real_embds is None:
            self.embds = nn.functional.normalize(
                    torch.randn(bank_size, vector_dim), dim=1)
            self.lure_embds = nn.functional.normalize(
                    self.embds + torch.randn(bank_size, vector_dim) * noise,
                    dim=1)
        else:
            assert real_embds.shape[0] == 2*bank_size
            assert real_embds.shape[1] == vector_dim
            self.embds = real_embds[::2]
            self.lure_embds = real_embds[1::2]
        
        # build fdata dict
        self.fdata = {}
        for order in range(1, 33):
            fname=self.root + os.sep + base_dir + os.sep + lag_set + os.sep + "order_{0}.txt".format(order)
            self.fdata[order] = np.genfromtxt(fname,dtype=int,delimiter=',')
        self.set_bins = np.array(self.check_files(self.SetName)) 
        if len(self.set_bins) != 192:
            raise ValueError('Set bin length is not the same as the stimulus set length (192)')
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        (repeat_list, lure_list, foil_list) = self.setup_list_permuted(self.set_bins)
        order = np.random.randint(1, 33)
        (type_code,ideal_resp,lag,fnames)= self.load_and_decode_order(repeat_list, lure_list,foil_list,lag_set=self.lag_set, order=order, base_dir = self.base_dir, stim_set=self.SetName)        
        vector = self.embds[fnames, :]
        lure_ids = np.where(ideal_resp == 1)[0]
        vector[lure_ids, :] = self.lure_embds[np.take(fnames, lure_ids), :] # replace the lures with lure_embds
        return dict(img=vector, target=ideal_resp)

    def check_files(self, SetName):
        """ 
        SetName should be something like "C" or "1"
        Checks to make sure there are the right #of images in the image directory
        Loads the lure bin ratings into the global set_bins list and returns this
        """
        
        
        #print(SetName)
        #print(P_N_STIM_PER_LIST)

        bins = []  # Clear out any existing items in the bin list

        # Load the bin file
        with open(self.root+"/Set"+str(SetName)+" bins.txt","r") as bin_file:
            reader=csv.reader(bin_file,delimiter='\t')
            for row in reader:
                if int(row[0]) > 192:
                    raise ValueError('Stimulus number ({0}) too large - not in 1-192 in binfile'.format(row[0]))
                if int(row[0]) < 1:
                    raise ValueError('Stimulus number ({0}) too small - not in 1-192 in binfile'.format(row[0]))
                bins.append(int(row[1]))
        if len(bins) != 192:
            raise ValueError('Did not read correct number of bins in binfile')

        # Check the stimulus directory
        img_list=glob.glob(self.root+"/Set " +str(SetName) + os.sep + '*.jpg')
        if len(img_list) < 384:
            raise ValueError('Not enough files in stimulus directory {0}'.format("Set " +str(SetName) + os.sep + '*.jpg'))
        for i in range(1,193):
            if not os.path.isfile(self.root+"/Set " +str(SetName) + os.sep + '{0:03}'.format(i) + 'a.jpg'):
                raise ValueError('Cannot find: ' + "Set " +str(SetName) + os.sep + '{0:03}'.format(i) + 'a.jpg')
            if not os.path.isfile(self.root+"/Set " +str(SetName) + os.sep + '{0:03}'.format(i) + 'b.jpg'):
                raise ValueError('Cannot find: ' + "Set " +str(SetName) + os.sep + '{0:03}'.format(i) + 'b.jpg')
        return bins

    def setup_list_permuted(self, set_bins):
        """
        set_bins = list of bin values for each of the 192 stimuli -- set specific

        Assumes check_files() has been run so we have the bin numbers for each stimulus

        Returns lists with the image numbers for each stimulus type (study, repeat...)
        in the to-be-used permuted order. Full 64 given for all.  This will get
        cut down and randomized in create_order()

        """


        


        # Figure the image numbers for the lure bins
        lure1=np.where(set_bins == 1)[0] + 1
        lure2=np.where(set_bins == 2)[0] + 1
        lure3=np.where(set_bins == 3)[0] + 1
        lure4=np.where(set_bins == 4)[0] + 1
        lure5=np.where(set_bins == 5)[0] + 1

        # Permute these
        lure1 = np.random.permutation(lure1)
        lure2 = np.random.permutation(lure2)
        lure3 = np.random.permutation(lure3)
        lure4 = np.random.permutation(lure4)
        lure5 = np.random.permutation(lure5)

        lures = np.empty(64,dtype=int)
        # Make the Lure list to go L1, 2, 3, 4, 5, 1, 2 ... -- 64 total of them (max)
        lure_count = np.zeros(5,dtype=int)
        nonlures = np.arange(1,193,dtype=int)
        for i in range(64):  
            if i % 5 == 0:
                lures[i]=lure1[lure_count[0]]
                lure_count[0]+=1
            elif i % 5 == 1:
                lures[i]=lure2[lure_count[1]]
                lure_count[1]+=1
            elif i % 5 == 2:
                lures[i]=lure3[lure_count[2]]
                lure_count[2]+=1
            elif i % 5 == 3:
                lures[i]=lure4[lure_count[3]]
                lure_count[3]+=1
            elif i % 5 == 4:
                lures[i]=lure5[lure_count[4]]
                lure_count[4]+=1
            nonlures=np.delete(nonlures,np.argwhere(nonlures == lures[i]))

        # Randomize the non-lures and split into 64-length repeat and foils
        nonlures=np.random.permutation(nonlures)
        foils = nonlures[0:64]
        repeats = nonlures[64:128]

        # At this point, we're full 64-item length lists for everything
        # break this down into the right size
        #repeatstim=repeats[0:set_size]
        #lurestim=lures[0:set_size]
        #foilstim=foils[0:set_size]

        # Our lures are still in L1, 2, 3, 4, 5, 1, 2, ... order -- fix that
        #lurestim=np.random.permutation(lurestim)


        return (repeats,lures,foils)
    
    def load_and_decode_order(self, repeat_list,lure_list,foil_list,
                          lag_set='AllShort_Set2',order=1,base_dir='LagGenerator',
                          stim_set='1'):
        """
        Loads the order text file and decodes this into a list of image names, 
         conditions, lags, etc.

        lag_set: Directory name with the order files
        order: Which order file to use (numeric index)
        base_dir: Directory that holds the set of lag sets
        stim_set = Set we're using (e.g., '1', or 'C')
        repeat_list,lure_list,foil_list: Lists (np.arrays actually) created by setup_list_permuted


        In the order files files we have 2 columns:
            1st column is the stimulus type + number:
            Offset_1R = 0; % 1-100 1st of repeat pair
            Offset_2R = 100; % 101-200  2nd of repeat pair
            Offset_1L = 200; % 201-300 1st of lure pair
            Offset_2L = 300; % 301-400 2nd of lure pair
            Offset_Foil = 400; % 401+ Foil

            2nd column is the lag + 500 (-1 for 1st and foil)

        Returns:
            lists / arrays that are all N-trials long

            type_code: 0=1st of repeat
                       1=2nd of repeat
                       2=1st of lure
                       3=2nd of lure
                       4=foil
            ideal_resp: 0=old
                        1=similar
                        2=new
            lag: Lag for this item (-1=1st/foil, 0=adjacent, N=items between)
            fnames: Actual filename of image to be shown
        """
        fdata = self.fdata[order]
        lag = fdata[:,1]
        lag[lag != -1] = lag[lag != -1] - 500

        type_code = fdata[:,0]//100  #Note, this works b/c we loaded the data as ints
        stim_index = fdata[:,0]-100*type_code
        ideal_resp = np.zeros_like(stim_index)
        ideal_resp[type_code==4]=2
        ideal_resp[type_code==0]=2
        ideal_resp[type_code==2]=2
        ideal_resp[type_code==1]=0
        ideal_resp[type_code==3]=1

        fnames=[]
        for i in range(len(type_code)):
            stimfile='UNKNOWN'
            if type_code[i]==0 or type_code[i]==1:
                stimfile=repeat_list[stim_index[i]]
            elif type_code[i]==2:
                stimfile=lure_list[stim_index[i]]
            elif type_code[i]==3:
                stimfile=lure_list[stim_index[i]]
            elif type_code[i]==4:
                stimfile=foil_list[stim_index[i]]
            fnames.append(stimfile - 1) # subtract 1 because file names start from 1 -> 192
        return (type_code,ideal_resp,lag,fnames)


@DATASETS.register_module
class MSTSaycamVectorDataset(MSTSynthVectorDataset):
    def __init__(
            self, root, saycam_root, list_file, which_model, 
            data_len=256*5000, seq_len=190,
            noise = 0.05,
            SetName = '1',
            lag_set='AllShort_Set2',
            base_dir='LagGenerator',
            bank_size=192,
            resamples = 1500):
        self.seq_len = seq_len
        self.data_len = data_len
        self.root = root
        self.saycam_root = saycam_root
        self.list_file = list_file
        self.which_model = which_model
        self.load_video_list()
        self.set_epoch(0)
        self.bank_size = bank_size
        self.resamples = resamples
        self.SetName = SetName
        self.lag_set = lag_set
        self.base_dir = base_dir
        
        # build fdata dict
        self.fdata = {}
        for order in range(1, 33):
            fname=self.root + os.sep + base_dir + os.sep + lag_set + os.sep + "order_{0}.txt".format(order)
            self.fdata[order] = np.genfromtxt(fname,dtype=int,delimiter=',')
        self.set_bins = np.array(self.check_files(self.SetName)) 
        if len(self.set_bins) != 192:
            raise ValueError('Set bin length is not the same as the stimulus set length (192)')
        # build embedding bank
        self.build_embedding_bank()
        
    def __len__(self):
        return self.data_len


    def build_embedding_bank(self):
        """
             Assume self.all_embds has shape [num_embeddings, 2, 128]
        """
        self.embds = []
        self.lure_embds = []
        embds_pos = []
        # resample embds to make sure that the foils are sufficiently different from one another
        for embd_id in range(self.bank_size):
            if embd_id == 0:
                pos_id = np.random.randint(self.all_embds.shape[0])
                self.embds.append(self.all_embds[pos_id, 0, :])
                self.lure_embds.append(self.all_embds[pos_id, 1, :])
            else: 
                i = 0
                while (i <= self.resamples):
                    i += 1
                    
                    pos_id = np.random.randint(self.all_embds.shape[0])
                    if pos_id in embds_pos:
                        continue
                    embds = np.stack(self.embds, axis=0)
                    dots = np.abs(np.dot(embds, self.all_embds[pos_id, 0, :]))
                    
                    if np.mean(dots) < 0.1:
                        break
                        
                    if i == self.resamples:
                        pos_id = np.random.choice([randint for randint in np.arange(self.all_embds.shape[0]) if randint not in embds_pos])
                self.embds.append(self.all_embds[pos_id, 0, :])
                self.lure_embds.append(self.all_embds[pos_id, 1, :])
                embds_pos.append(pos_id)
                     
        self.embds = np.stack(self.embds, axis=0)
        self.lure_embds = np.stack(self.lure_embds, axis=0)
        
    def load_video_list(self):
        with open(self.list_file, 'r') as fin:
            all_lines = fin.readlines()
            self.video_list = [_line[:-1] for _line in all_lines]
        
        
    def load_embds(self):
        possible_sta_pos = []
        embds = []
        curr_len = 0

        def _get_offset_num(name):
            return int(name.split('_')[1])
        def _get_aug_num(name):
            return int(name.split('_')[3].split('.')[0])

        for video_name in self.video_list:
            
            
            embd_dir = os.path.join(self.saycam_root, self.which_model, video_name)
            
            embd_files = os.listdir(embd_dir)
            
            one_embd_file = np.random.choice(embd_files)
            offset_num = _get_offset_num(one_embd_file)
            aug_num = _get_aug_num(one_embd_file)
            filtered_embd_files = filter(
                    lambda x: (_get_offset_num(x) == offset_num) \
                              & (_get_aug_num(x) != aug_num),
                    embd_files, 
                    )


            another_embd_file = np.random.choice(list(filtered_embd_files))
            one_embd = np.load(os.path.join(embd_dir, one_embd_file))
            another_embd = np.load(os.path.join(embd_dir, another_embd_file))
            an_embd = np.stack([one_embd, another_embd], axis=1)
            embds.append(an_embd)
            possible_sta_pos.extend(
                    list(range(curr_len, 
                               curr_len + len(one_embd) - self.seq_len+1)))
            curr_len += len(one_embd)


        self.all_embds = np.concatenate(embds, axis=0)
        self.all_sta_pos = np.random.choice(
                possible_sta_pos, self.data_len, replace=True)
        
        
    def set_epoch(self, epoch):
        np.random.seed(epoch)
        self.load_embds()
