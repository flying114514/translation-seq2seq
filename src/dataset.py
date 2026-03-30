import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import config


# 定义Dataset
class TranslationDataset(Dataset):
    def __init__(self, path):
        self.data = pd.read_json(path, lines=True, orient='records').to_dict(orient='records')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_tensor = torch.tensor(self.data[index]['zh'], dtype=torch.long)
        target_tensor = torch.tensor(self.data[index]['en'], dtype=torch.long)
        return input_tensor, target_tensor



# 提供自定义的collate_fn函数,用于处理不同长度的输入和目标序列,将它们填充到相同的长度
# 由于我们现在先分批,后填充
def collate_fn(batch):
    # batch是一个二元组列表,每个二元组包含一个输入张量和一个目标张量,比如[(input_tensor1, target_tensor1), (input_tensor2, target_tensor2), ...]
    input_tensors = [item[0] for item in batch]
    target_tensors = [item[1] for item in batch]

    # 使用pad_sequence函数将输入张量和目标张量填充到相同的长度,并返回填充后的张量
    input_tensor = pad_sequence(input_tensors, batch_first=True, padding_value=0, padding_side='right')
    output_tensor = pad_sequence(target_tensors, batch_first=True, padding_value=0, padding_side='right')

    return input_tensor, output_tensor

# 提供获取dataloader方法
def get_dataloader(train=True):
    path = config.PROCESSED_DATA_DIR / ('train.jsonl' if train else 'test.jsonl')
    dataset = TranslationDataset(path)
    return DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

if __name__ == '__main__':
    train_dataloader = get_dataloader()
    test_dataloader = get_dataloader(train=False)
    print(len(train_dataloader))
    print(len(test_dataloader))

    for input_tensor, target_tensor in train_dataloader:
        # 每一批划分的长度是不同的
        print(input_tensor.shape)
        print(target_tensor.shape)
        print("================")
