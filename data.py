import torch
import json

class PoseDataset(torch.utils.data.Dataset):
    def __init__(self, json_file):
        data = []

        with open(json_file) as file:
            for line in file.readlines():
                data += json.loads(line)
        
        data = map(lambda x: list(x.values()), data)
        data = list(data)

        self.data = torch.tensor(data, dtype=torch.float)

        # Normalizaiton
        mean = (torch.mean(self.data, dim=0))
        std = (torch.std(self.data, dim=0))

        # print(mean)
        print(self.data.mean())
        torch.save(mean, 'mean.pt')
        torch.save(std, 'std.pt')

        self.data = (self.data - mean) / std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        return sample
