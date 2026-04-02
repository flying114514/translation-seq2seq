import torch
from torch import nn
import config


class TranslationEncoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)
        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

    def forward(self, x):
        # x.shape: [batch_size, seq_len]
        embed = self.embedding(x)
        # embed.shape: [batch_size, seq_len, embedding_dim}
        output, _ = self.gru(embed)
        # output.shape: [batch_size, seq_len, hidden_size]

        # 每批样本的最后一个非pad的时间步的输出作为文本的表示
        # lengths是一批样本中每个文本的实际长度,也就是非pad的时间步的数量,它的形状是[batch_size]
        lengths = (x != self.embedding.padding_idx).sum(dim=1)
        last_hidden_state = output[torch.arange(output.shape[0]), lengths - 1]
        # last_hidden_state.shape: [batch_size, hidden_size]
        return last_hidden_state


class TranslationDecoder(nn.Module):
    def __init__(self, vocab_size, padding_index):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=config.EMBEDDING_DIM,
                                      padding_idx=padding_index)

        self.gru = nn.GRU(input_size=config.EMBEDDING_DIM,
                          hidden_size=config.HIDDEN_SIZE,
                          batch_first=True)

        self.linear = nn.Linear(config.HIDDEN_SIZE, vocab_size)

    def forward(self, x, hidden_0):
        # 传进来的英语单词智能是一个,所以seq_len是1
        # x.shape: [batch_size, 1]
        # hidden_0是编码器传过来的文本表示,它的形状是[1, batch_size, hidden_size],其中1是因为GRU的num_layers是1,如果是多层的话就是num_layers
        # hidden.shape: [1, batch_size, hidden_size]
        embed = self.embedding(x)
        # x经过编码后,出现了embedding_dim这个新维度
        # embed.shape: [batch_size, 1, embedding_dim]
        # 传入GRU,得到输出和新的隐藏状态
        output, hidden_n = self.gru(embed, hidden_0)
        # output.shape: [batch_size, 1, hidden_size]
        # hidden_n.shape: [1, batch_size, hidden_size]
        output = self.linear(output)
        # output在线性层和词表对应
        # output.shape: [batch_size, 1, vocab_size]
        return output, hidden_n


class TranslationModel(nn.Module):
    def __init__(self, zh_vocab_size, en_vocab_size, zh_padding_index, en_padding_index):
        super().__init__()
        self.encoder = TranslationEncoder(zh_vocab_size, zh_padding_index)
        self.decoder = TranslationDecoder(en_vocab_size, en_padding_index)
