import os
from argparse import ArgumentParser
import shutil
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from bcresnet import BCResNets
from utils2 import DownloadDataset, Padding, Preprocess, SpeechCommand, SplitDataset
import random
import torch.nn as nn

label_dict = {
    "HiCellico": 0,
    "ZoomIn": 1,
    "ZoomOut": 2,
    "detectObject": 3,
    "ReadText": 4,
    "unknown" : 5,
    "_background_noise_": 6,
    "_silence_" : 7
}
# 새로운 데이터셋 클래스를 정의합니다.
class CustomSpeechDataset(Dataset):
    def __init__(self, root_dir, noise_dir=None, transform=None, sample_rate=16000):
        self.transform = transform
        self.data_list, self.labels = self._scan_audio_files(root_dir)
        self.resample = torchaudio.transforms.Resample(orig_freq=44100, new_freq=sample_rate)
        self.sample_rate = sample_rate
        self.duration = 2
        if noise_dir:
            self.background_noise = [
                torchaudio.load(file_name)[0] for file_name in glob(noise_dir + "/*.wav")
            ]
        else:
            self.background_noise = []

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        audio_path = self.data_list[idx]
        sample, orig_sample_rate = torchaudio.load(audio_path)
        label = self.labels[idx]
        if orig_sample_rate != self.sample_rate:
            sample = self.resample(sample)
        if sample.shape[0] > 1:  # 스테레오인 경우 모노로 변환
            sample = torch.mean(sample, dim=0, keepdim=True)
        aa = sample
        sample = self._pad_or_trim(sample)
        #print(audio_path)
        #print('aa', sample.shape)
        if self.transform:
            sample = self.transform(sample)  # Add channel dimension and labels
        #print('bb', sample.shape)
        #print(sample.shape, aa.shape)
        return sample, label

    def _scan_audio_files(self, root_dir):
        audio_paths, labels = [], []
        for path, _, files in sorted(os.walk(root_dir, followlinks=True)):
            for filename in files:
                if not (filename.endswith(".mp3") or filename.endswith(".wav")):
                    continue
                class_name = os.path.basename(path)
                if class_name not in label_dict:
                    continue
                audio_paths.append(os.path.join(path, filename))
                labels.append(label_dict[class_name])
        for a, b in zip(audio_paths, labels):        
            print("check", a, b)
        print(len(audio_paths))
        return audio_paths, labels

    def _pad_or_trim(self, sample):
        num_samples = int(self.sample_rate * self.duration)
        if sample.shape[1] > num_samples:
            sample = sample[:, :num_samples]
        elif sample.shape[1] < num_samples:
            padding = num_samples - sample.shape[1]
            pad_left = padding // 2
            pad_right = padding - pad_left
            sample = torch.nn.functional.pad(sample, (pad_left, pad_right))
        return sample

    def _add_background_noise(self, sample):
        if self.background_noise:
            noise = random.choice(self.background_noise)
            #noise2 = random.choice(self.background_noise)
            #noise = torch.cat((noise1, noise2), dim=1)  # 두 개의 1초짜리 노이즈를 이어붙임
            
            noise = self._pad_or_trim(noise)
            #print(noise.shape, sample.shape)
            noise_amp = np.random.uniform(0, 0.5)
            sample = sample + noise_amp * noise
            sample = torch.clamp(sample, -1.0, 1.0)
        return sample

# 기존 Trainer 클래스 수정
class Trainer:
    def __init__(self):
        parser = ArgumentParser()
        parser.add_argument(
            "--data_dir", required=True, help="Path to the dataset directory"
        )
        parser.add_argument(
            "--noise_dir", required=True, help="Path to the background noise directory"
        )
        parser.add_argument(
            "--tau", default=1, help="model size", type=float, choices=[1, 1.5, 2, 3, 6, 8]
        )
        parser.add_argument("--gpu", default=0, help="gpu device id", type=int)
        args = parser.parse_args()
        self.__dict__.update(vars(args))
        self.device = torch.device("cuda:%d" % self.gpu if torch.cuda.is_available() else "cpu")
        self._load_data()
        self._load_model()

    def __call__(self):
        total_epoch = 200
        warmup_epoch = 5
        init_lr = 3e-3
        lr_lower_limit = 0

        optimizer = torch.optim.SGD(self.model.parameters(), lr=0, weight_decay=1e-3, momentum=0.9)
        n_step_warmup = len(self.train_loader) * warmup_epoch
        total_iter = len(self.train_loader) * total_epoch
        iterations = 0

        for epoch in range(total_epoch):
            self.model.train()
            for sample in tqdm(self.train_loader, desc="epoch %d, iters" % (epoch + 1)):
                iterations += 1
                if iterations < n_step_warmup:
                    lr = init_lr * iterations / n_step_warmup
                else:
                    lr = lr_lower_limit + 0.5 * (init_lr - lr_lower_limit) * (
                        1
                        + np.cos(
                            np.pi * (iterations - n_step_warmup) / (total_iter - n_step_warmup)
                        )
                    )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                inputs, labels = sample
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                inputs = self.preprocess_train(inputs, labels, augment=True)
                outputs = self.model(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                self.model.zero_grad()

                # 마지막 3개 샘플을 저장하기 위한 코드 추가
                self.last_samples.append((inputs.cpu(), labels.cpu()))
                if len(self.last_samples) > 3:  # 최대 3개까지만 유지
                    self.last_samples.pop(0)

            print("cur lr check ... %.4f" % lr)
            with torch.no_grad():
                self.model.eval()
                valid_acc = self.Test(self.valid_dataset, self.valid_loader, augment=True)
                print("valid acc: %.3f" % (valid_acc))
            final_model_path = f"finetune/bcresnet_finetune_{epoch}.pt"
            torch.save(self.model.state_dict(), final_model_path)

        self._save_last_samples()
        test_acc = self.Test(self.test_dataset, self.test_loader, augment=False)
        print("test acc: %.3f" % (test_acc))
        print("End.")
        
    def _save_last_samples(self):
        save_dir = "last_samples"
        os.makedirs(save_dir, exist_ok=True)
        for i, (inputs, labels) in enumerate(self.last_samples):
            file_name = f"sample_{i + 1}.wav"
            file_path = os.path.join(save_dir, file_name)

            # 각 샘플을 저장합니다.
            torchaudio.save(file_path, inputs[0], 16000)  # 첫 번째 채널만 저장
            print(f"Saved: {file_path}")

    def Test(self, dataset, loader, augment):
        true_count = 0.0
        num_testdata = float(len(dataset))
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            inputs = self.preprocess_test(inputs, labels=labels, is_train=False, augment=augment)
            outputs = self.model(inputs)
            prediction = torch.argmax(outputs, dim=-1)
            true_count += torch.sum(prediction == labels).detach().cpu().numpy()
        acc = true_count / num_testdata * 100.0
        return acc

    def _load_data(self):
        print("Loading custom dataset ...")

        train_dir = os.path.join(self.data_dir, 'train')
        valid_dir = os.path.join(self.data_dir, 'valid')
        test_dir = os.path.join(self.data_dir, 'valid')

        self.train_dataset = CustomSpeechDataset(train_dir, self.noise_dir, transform=None)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=8, shuffle=True, num_workers=0, drop_last=False
        )
        self.valid_dataset = CustomSpeechDataset(valid_dir, self.noise_dir, transform=None)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=8, num_workers=0)
        self.test_dataset = CustomSpeechDataset(test_dir, self.noise_dir, transform=None)
        self.test_loader = DataLoader(self.test_dataset, batch_size=8, num_workers=0)

        print(
            "check num of data train/valid/test %d/%d/%d"
            % (len(self.train_dataset), len(self.valid_dataset), len(self.test_dataset))
        )

        specaugment = self.tau >= 1.5
        frequency_masking_para = {1: 0, 1.5: 1, 2: 3, 3: 5, 6: 7, 8: 7}

        self.preprocess_train = Preprocess(
            self.noise_dir,
            self.device,
            specaug=specaugment,
            frequency_masking_para=frequency_masking_para[self.tau],
        )
        self.preprocess_test = Preprocess(self.noise_dir, self.device)

    def _load_model(self):
        print("model: BC-ResNet-%.1f" % (self.tau))
        self.model = BCResNets(int(self.tau * 8)).to(self.device)
        # Pre-trained model 불러오기
        pre_trained_model_path = "bcresnet2_51.pt"
        self.model.load_state_dict(torch.load(pre_trained_model_path))
        self.model.classifier[-1] = nn.Conv2d(self.model.c[-1], 8, 1).to(self.device)
        for param in self.model.parameters():
            param.requires_grad = True

        # 마지막 분류 레이어의 파라미터만 학습 가능하게 설정
        for param in self.model.classifier.parameters():
            param.requires_grad = True

if __name__ == "__main__":
    _trainer = Trainer()
    _trainer()