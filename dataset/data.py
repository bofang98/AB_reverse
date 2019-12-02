import os
import torch.utils.data as data
import cv2
import sys
sys.path.append('..')
import random
from config import params
import skvideo.io
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import numpy as np
import pandas as pd
import argparse


envs = os.environ

class ReverseDataSet(data.Dataset):
    def __init__(self, root, mode="train"):

        self.transforms = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.RandomCrop(112),
            transforms.ToTensor()])


        self.root = root
        self.mode = mode
        self.videos = []
        self.labels = []
        self.toPIL = transforms.ToPILImage()
        self.split = '1'

        class_idx_path = os.path.join(root, 'split', 'classInd.txt')
        self.class_idx2label = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(0)[1]
        self.class_label2idx = pd.read_csv(class_idx_path, header=None, sep=' ').set_index(1)[0]
        if self.mode == 'train':
            train_split_path = os.path.join(root, 'split', 'trainlist0' + self.split + '.txt')
            self.train_split = pd.read_csv(train_split_path, header=None, sep=' ')[0]
        else:
            test_split_path = os.path.join(root, 'split', 'testlist0' + self.split + '.txt')
            self.test_split = pd.read_csv(test_split_path, header=None)[0]
        print('Use split' + self.split)

    def loadcvvideo(self, fname, count_need=16):
        fname = os.path.join(self.root, 'video', fname)
        capture = cv2.VideoCapture(fname)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if count_need == 0:
            count_need = frame_count
        start = np.random.randint(0, frame_count - count_need + 1)
        buffer = []
        count = 0
        retaining = True
        sample_count = 0

        while (sample_count < count_need and retaining):
            retaining, frame = capture.read()

            if retaining is False:
                count += 1

                break
            if count >= start:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
               # frame = cv2.resize(frame, (171, 128))
                buffer.append(frame)
                sample_count = sample_count + 1
            count += 1

        capture.release()

        return buffer, retaining

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_split)-1
        else:
            return len(self.test_split)-1

    def __getitem__(self, index):
        if self.mode == 'train':
            videoname = self.train_split[index]
        else:
            videoname = self.test_split[index]

        if self.mode == 'train':
            videodata, retrain = self.loadcvvideo(videoname, count_need=16)
            #print('videodata: ', np.array(videodata).shape, ' retrain: ', retrain)
            while retrain == False or len(videodata) < 16:
                print('reload')
                index = np.random.randint(self.__len__())

                videoname = self.train_split[index]
                videodata, retrain = self.loadcvvideo(videoname, count_need=16)

            videodata = self.randomflip(videodata)

            video_clips = []
            seed = random.random()

            for frame in videodata:
                random.seed(seed)
                frame = self.toPIL(frame)
                frame = self.transforms(frame)
                video_clips.append(frame)
            # video_clips, which is a list, contains 16 tensors
            #clip = torch.stack(video_clips).permute(1, 0, 2, 3)
            #clip = torch.stack(video_clips)
            part = []
            divide_frame = random.randint(0, 14)
            for j in range(divide_frame + 1, 16):
                part.append(video_clips[j])
            #print('part_1: ', part)
            for i in range(divide_frame + 1):
                part.append(video_clips[i])
            #print('part_2: ', part)
            clip = torch.stack(part).permute(1, 0, 2, 3)
            label = divide_frame

        elif self.mode == 'test':
            videodata, retrain = self.loadcvvideo(videoname, count_need=0)
            while retrain == False or len(videodata) < 16:
                print('reload')
                index = np.random.randint(self.__len__())

                videoname = self.test_split[index]
                videodata, retrain = self.loadcvvideo(videoname, count_need=16)
            clip = self.gettest(videodata)
            label = self.class_label2idx[videoname[:videoname.find('/')]]

        return clip, label

    def randomflip(self, buffer):
        if np.random.random() <0.5:
            for i,frame in enumerate(buffer):

                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def gettest(self, videodata):
        length = len(videodata)

        all_clips = []

        for i in np.linspace(8, length-8, 10):
                clip_start = int(i - 8)
                clip = videodata[clip_start: clip_start + 16]
                trans_clip = []
                    # fix seed, apply the sample `random transformation` for all frames in the clip
                seed = random.random()
                for frame in clip:
                        random.seed(seed)
                        frame = self.toPIL(frame) # PIL image
                        frame = self.transforms(frame) # tensor [C x H x W]
                        trans_clip.append(frame)
                    # (T x C X H x W) to (C X T x H x W)
                clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
                all_clips.append(clip)
        print('all_clips:', all_clips)
        return torch.stack(all_clips)


if __name__ == '__main__':

    data = ReverseDataSet(params['dataset'], mode='test')
    train_data = DataLoader(data, batch_size=8, num_workers=4, shuffle=True)

    for i, (clip, label) in enumerate(train_data):
        print(i, ':')
        print('clip: ', clip.shape)
        print('label: ', label)
        print('------------------------------------------------')
        if i == 0:
            break

