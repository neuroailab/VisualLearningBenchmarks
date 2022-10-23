import os, sys
import numpy as np
from PIL import Image
import argparse
import pdb
import torch
import torchvision.transforms as transforms
import pickle
from tqdm import tqdm

USER_NAME = os.getlogin()
IMGNT_RAW_FOLDER = f'/data5/{USER_NAME}/Dataset/imagenet_raw/train/'
FACE_RAW_FOLDER = f'/data5/{USER_NAME}/Dataset/vgg_face2/jpgs/train/'
EVAL_BATCH_SIZE = int(os.environ.get('EVAL_BATCH_SIZE', 100))


class ImgNtImgSmplr(object):
    def __init__(
            self, num_classes=30, 
            num_imgs_per_class=100,
            num_eval_imgs_per_class=20,
            random_seed=0,
            raw_folder=IMGNT_RAW_FOLDER):
        self.num_classes = num_classes
        self.num_imgs_per_class = num_imgs_per_class
        self.num_eval_imgs_per_class = num_eval_imgs_per_class
        self.raw_folder = raw_folder

        np.random.seed(random_seed)
        self.__sample_classes()
        self.__sample_imgs()
        self.__get_imgs()

    def __sample_classes(self):
        all_classes = os.listdir(self.raw_folder)
        all_classes.sort()
        self.sample_classes = np.random.choice(
                all_classes, self.num_classes, 
                replace=False)

    def __sample_imgs(self):
        train_imgs = []
        eval_imgs = []
        for each_class in self.sample_classes:
            class_path = os.path.join(self.raw_folder, each_class)
            all_imgs = os.listdir(class_path)
            all_imgs.sort()
            all_imgs = [os.path.join(class_path, _img) for _img in all_imgs]
            all_sampled_imgs = np.random.choice(
                    all_imgs, 
                    self.num_imgs_per_class + self.num_eval_imgs_per_class,
                    replace=False)
            train_imgs.append(all_sampled_imgs[:self.num_imgs_per_class])
            eval_imgs.append(all_sampled_imgs[self.num_imgs_per_class:])
        self.train_img_paths = np.concatenate(train_imgs)
        self.eval_img_paths = np.concatenate(eval_imgs)

    def __load_imgs_from_paths(self, paths):
        img_norm_cfg = dict(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225])
        prep_trans = transforms.Compose(
                [transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize(**img_norm_cfg),
                 ])
        all_imgs = []
        for img_path in tqdm(paths):
            img = Image.open(img_path).convert('RGB')
            img = prep_trans(img)
            if img.shape[0] == 1:
                img = img.repeat(3, 1, 1)
            all_imgs.append(img)
        imgs = torch.stack(all_imgs, axis=0)
        return imgs

    def __get_imgs(self):
        print('Getting Performance Images')
        self.train_imgs = self.__load_imgs_from_paths(self.train_img_paths)
        self.train_imgs = self.train_imgs.cuda()
        print('Getting Performance Validation Images')
        self.eval_imgs = self.__load_imgs_from_paths(self.eval_img_paths)
        self.eval_imgs = self.eval_imgs.cuda()

    def get_resp(self, model, imgs):
        model.eval()
        batch_size = EVAL_BATCH_SIZE
        i = 0
        all_resp = []
        while i < imgs.size()[0]:
            end_idx = min(imgs.size()[0], i+batch_size)
            cur_imgs = imgs[i : end_idx]
            resp = model(cur_imgs, mode='test')['embd']
            resp = resp.detach().numpy()
            all_resp.extend(resp)
            i += batch_size
        all_resp = np.stack(all_resp, axis=0)
        return all_resp
        
        
    def eval_model(self, model):
        train_resp = self.get_resp(model, self.train_imgs)
        eval_resp = self.get_resp(model, self.eval_imgs)
        
        dp_results = np.matmul(eval_resp, np.transpose(train_resp, [1, 0]))
        pred_nn = np.argmax(dp_results, axis=1)
        right_num = 0
        
        for idx in range(len(pred_nn)):
            pred_label = self.train_img_paths[pred_nn[idx]].split('/')[-2]
            corr_label = self.eval_img_paths[idx].split('/')[-2]
            if pred_label == corr_label:
                right_num += 1
        accuracy = right_num * 1.0 / len(self.eval_img_paths)
        return accuracy


def test_get_imgnt_imgs():
    imgnt_img_smplr = ImgNtImgSmplr()
    import build_response
    args = build_response.get_simclr_r18_args()
    model_builder = build_response.ModelBuilder(args)
    model_builder.load_weights()
    model = model_builder.model
    model.cuda()
    accuracy = imgnt_img_smplr.eval_model(model)
    print(accuracy)


if __name__ == '__main__':
    test_get_imgnt_imgs()
