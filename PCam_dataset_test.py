import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Custom Dataset Class
class PCamDataset(Dataset):
    def __init__(self, x_file, y_file):
        # Load the h5py files
        self.x_file = h5py.File(x_file, 'r')
        self.y_file = h5py.File(y_file, 'r')
        
        # Get the number of samples
        self.num_samples = len(self.x_file['x'])  # x 데이터셋 크기 (배열 길이)
        
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Read the data and label one by one
        x_data = np.array(self.x_file['x'][idx], dtype=np.float32)  # (96, 96, 3) -> 96x96px RGB 이미지
        y_data = np.array(self.y_file['y'][idx], dtype=np.long)  # 정수형 라벨
        
        # Convert to PyTorch tensors
        x_data = torch.tensor(x_data)
        y_data = torch.tensor(y_data)
        
        return x_data, y_data
    
    def __del__(self):
        # Close the h5py files when the dataset is deleted
        self.x_file.close()
        self.y_file.close()

# File paths
x_file = r"C:\PCam\camelyonpatch_level_2_split_train_x.h5"
y_file = r"C:\PCam\camelyonpatch_level_2_split_train_y.h5"

# Create Dataset and DataLoader
dataset = PCamDataset(x_file, y_file)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


for x_batch, y_batch in dataloader:
    print("x_batch shape:", x_batch.shape)  # 이미지 배치 크기 출력
    print("y_batch shape:", y_batch.shape)  # 라벨 배치 크기 출력
    print("x_batch:", x_batch)  # x_batch 출력
    print("y_batch:", y_batch)  # y_batch 출력
    break  # 첫 번째 배치만 출력 후 종료
