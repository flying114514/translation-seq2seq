import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_dataloader
from src.model import TranslationModel
from src.tokenizer import ChineseTokenizer, EnglishTokenizer
import config


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    total_loss = 0
    model.train()
    for inputs, targets in tqdm(dataloader, desc='训练中'):
        encoder_inputs = inputs.to(device)  # inputs.shape:[batch_size, src_seq_len]
        targets = targets.to(device)  # target.shape:[batch_size, tgt_seq_len]
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        # decoder_inputs.shape: [batch_size, tgt_seq_len-1]
        # decoder_targets.shape: [batch_size, tgt_seq_len-1]

        # 前向传播
        # 编码
        context_vector = model.encoder(encoder_inputs)
        # context_vector.shape: [batch_size, hidden_size]

        decoder_outputs = []
        # 解码
        decoder_hidden = context_vector.unsqueeze(0)  # decoder_hidden_0.shape: [1, batch_size, hidden_size]
        sql_len = decoder_inputs.shape[1]
        for i in range(sql_len):
            decoder_input = decoder_inputs[:, i].unsqueeze(1)  # decoder_input.shape: [batch_size, 1]
            decoder_output, decoder_hidden = model.decoder(decoder_input, decoder_hidden)
            # decoder_output.shape: [batch_size,1 , vocab_size]
            decoder_outputs.append(decoder_output)

        # loss
        # decoder_outputs.shape: [tensor([batch_size,1,vocab_size])] -> [batch_size * (tgt_seq_len-1), vocab_size]
        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = decoder_outputs.reshape(-1, decoder_outputs.shape[-1])
        # decoder_targets.shape: [batch_size, tgt_seq_len-1] -> [batch_size * (tgt_seq_len-1)]
        decoder_targets = decoder_targets.reshape(-1)

        loss = loss_fn(decoder_outputs, decoder_targets)

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def train():
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据
    dataloader = get_dataloader()
    # 分词器
    zh_Tokenizer = ChineseTokenizer.from_vocab(config.MODELS_DIR / 'zh_vocab.txt')
    en_Tokenizer = EnglishTokenizer.from_vocab(config.MODELS_DIR / 'en_vocab.txt')
    # 模型
    model = (TranslationModel(zh_Tokenizer.vocab_size, en_Tokenizer.vocab_size,
                              zh_Tokenizer.pad_token_index, en_Tokenizer.pad_token_index)
             .to(device))
    # 损失函数
    loss_fn = torch.nn.CrossEntropyLoss()
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # TensorBoard Writer
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    best_loss = float('inf')
    for epoch in range(1, config.EPOCHS + 1):
        print("=" * 10, f"Epoch:{epoch}", "=" * 10)
        # 训练一个epoch
        loss = train_one_epoch(model, dataloader, loss_fn, optimizer, device)
        print(f"loss:{loss:.4f}")

        # 记录损失数据
        writer.add_scalar('loss', loss, epoch)
        # 保存模型
        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), config.MODELS_DIR / 'best.pth')
            print("模型保存成功")

    writer.close()


if __name__ == "__main__":
    train()
