import torch
import timm
import numpy as np
from torch import nn
from pascal_voc_multi_label_dataset import PascalVOCMultiLabelDataset
from torchvision import transforms
import torchvision

data_transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

voc_val = torchvision.datasets.VOCDetection(root='./data', year='2012', image_set='val', download=True, transform=data_transform)
val_dataset = PascalVOCMultiLabelDataset(voc_val)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

num_classes = 20  # 변경하려는 클래스 수에 따라 이 값을 수정하세요.


model = timm.create_model('swin_base_patch4_window12_384', pretrained=False)
# num_classes = 20
# 출력 레이어 수정하기
in_features = model.head.in_features # 기존 출력 레이어의 input feature 개수
out_features = num_classes # 새로운 출력 레이어의 output feature 개수
model.head = torch.nn.Linear(in_features, out_features)

# # EfficientNet-B2 모델 생성
# model = timm.create_model('efficientnet_b2', pretrained=False)  # 사전 훈련된 가중치를 사용합니다.

# # 출력 레이어를 멀티 레이블 작업에 맞게 수정
# model.classifier = nn.Sequential(
#     nn.Linear(model.classifier.in_features, num_classes),
#     nn.Sigmoid()
# )



# 모델 가중치 불러오기
model.load_state_dict(torch.load('models/best_model_swin.pth'))

# GPU 사용 가능한 경우에 사용
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 모델을 평가 모드로 설정
model.eval()

def accuracy_metric(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    y_pred = (y_pred > 0.5).astype(int)  # 임계값 0.5를 기준으로 이진 분류 수행
    return np.mean(y_true == y_pred)

# 검증 세트에 대한 정확도 계산
val_accuracy = 0
n_samples = 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        targets = targets.cpu().numpy()
        outputs = outputs.cpu().numpy()
        val_accuracy += accuracy_metric(targets, outputs) * targets.shape[0]
        n_samples += targets.shape[0]

val_accuracy /= n_samples
print(f"Validation accuracy: {val_accuracy:.4f}")
