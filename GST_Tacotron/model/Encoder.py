import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ConvEncoder(nn.Module):
    def __init__(
        self,
        num_vocab=40,
        embed_dim=256,
        conv_layers=3,
        conv_channels=256,
        conv_kernel_size=5,
    ):
        super().__init__()
        # 文字埋め込み
        self.embed = nn.Embedding(num_vocab, embed_dim, padding_idx=0)

        # 1次元畳み込みの重ね合わせ：局所的な依存関係のモデル化
        self.convs = nn.ModuleList()
        for layer in range(conv_layers):
            in_channels = embed_dim if layer == 0 else conv_channels
            self.convs += [
                nn.Conv1d(
                    in_channels,
                    conv_channels,
                    conv_kernel_size,
                    padding=(conv_kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.Dropout(0.5),
            ]
        self.convs = nn.Sequential(*self.convs)

    def forward(self, seqs):
        emb = self.embed(seqs)
        # 1 次元畳み込みと embedding では、入力のサイズが異なるので注意
        out = self.convs(emb.transpose(1, 2)).transpose(1, 2)
        return out

class Encoder(ConvEncoder):
    def __init__(
        self,
        num_vocab=40,
        embed_dim=512,
        hidden_dim=256,
        conv_layers=3,
        conv_channels=512,
        conv_kernel_size=5,
    ):
        super().__init__(
            num_vocab, embed_dim, conv_layers, conv_channels, conv_kernel_size
        )
        # 双方向 LSTM による長期依存関係のモデル化
        self.blstm = nn.LSTM(
            conv_channels, hidden_dim // 2, 1, batch_first=True, bidirectional=True
        )

    def forward(self, seqs, in_lens):
        emb = self.embed(seqs)
        # 1 次元畳み込みと embedding では、入力のサイズ が異なるので注意
        out = self.convs(emb.transpose(1, 2)).transpose(1, 2)

        # 双方向 LSTM の計算
        out = pack_padded_sequence(out, in_lens, batch_first=True)
        out, _ = self.blstm(out)
        out, _ = pad_packed_sequence(out, batch_first=True)
        return out
