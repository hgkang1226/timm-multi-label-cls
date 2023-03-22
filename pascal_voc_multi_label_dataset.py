import torch
from torch.utils.data import Dataset


class PascalVOCMultiLabelDataset(Dataset):
    def __init__(self, voc_dataset):
        self.voc_dataset = voc_dataset
        self.class_names = [
            "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
        ]  # 클래스 이름 직접 정의

    def __getitem__(self, index):
        image, target = self.voc_dataset[index]
        labels = target['annotation']['object']
        
        # 멀티 레이블 생성
        multi_label = torch.zeros(len(self.class_names), dtype=torch.float32)
        for label in labels:
            class_name = label['name']
            class_index = self.class_names.index(class_name)
            multi_label[class_index] = 1.0
        
        return image, multi_label

    def __len__(self):
        return len(self.voc_dataset)
