import torch
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self,data, labels):
        self.data=data
        self.labels=labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample=self.data[idx]
        label=self.labels[idx]
        return sample, label


data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
print(data.shape)
labels=torch.tensor([0, 1, 0, 1])
print(labels.shape)

dataset=CustomDataset(data,labels)
dataloader=DataLoader(dataset, batch_size=2, shuffle=True)

for _, (data, label) in enumerate(dataloader):
    print(_)
    print(data)
    print(label)



