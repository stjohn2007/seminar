import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

class ReferenceEncoder(nn.Module):
    # input : (B, 1, seq_len, n_mel)
    # output : (1, B, 128)

    def __init__(
        self,
        n_mel = 80,
        conv_channels1 = 32,
        conv_channels2 = 32,
        conv_channels3 = 64,
        conv_channels4 = 64,
        conv_channels5 = 128,
        conv_channels6 = 128,
        n_unit = 128
    ):
        super().__init__()
        conv_channels_list = [
            conv_channels1,
            conv_channels2,
            conv_channels3,
            conv_channels4,
            conv_channels5,
            conv_channels6,
        ]
        
        self.convs = nn.ModuleList()
        for layer in range(6):
            n_mel = (n_mel + 1) // 2

            in_channels = 1 if layer == 0 else conv_channels_list[layer-1]
            out_channels = conv_channels_list[layer]
            self.convs += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            ]
        self.convs = nn.Sequential(*self.convs)
        
        n_mel *= conv_channels_list[-1]
        self.gru = nn.GRU(n_mel, n_unit, batch_first=True)
        
    def forward(self, seqs):
        out = self.convs(seqs) # (B, 1, seq_len, n_mel) -> (B, channels, new_seq_len, new_n_mel)
        out = torch.squeeze(out.reshape(out.shape[0], 1, out.shape[2], -1)) # (B, channels, new_seq_len, new_n_mel) -> (B, new_seq_len, channels * new_n_mel)
        out = self.gru(out) # (B, new_seq_len, channels * new_n_mel) -> (1, B, n_unit) 最終隠れ層をembeddingとして採用する
        return out[0], out[1]


class StyleTokenLayer(nn.Module):
    # input : (B, 1, 128)
    # output : (B, 256)

    def __init__(
        self,
        ref_emb_size = 128,
        emb_size = 256,
        n_tokens = 10,
        device = "cuda"
    ):
        super().__init__()
        self.queryL = nn.Linear(ref_emb_size, emb_size)
        self.keyL = nn.Linear(emb_size, emb_size)
        self.valueL = nn.Linear(emb_size, emb_size)
        self.tanh = nn.Tanh()
        self.emb_size = emb_size,
        self.softmax = nn.Softmax(dim=1) # n_token方向にsoftmaxを取りたい
        self.tokens = torch.randn(n_tokens, emb_size).to(device)
    
    def forward(self, query=None, token_num=0):
        if query == None: #inference
            return self.tokens[token_num]
        
        query = self.queryL(query) # (B, 1, 128) -> (B, 1, 256)
        tokens = self.tanh(self.tokens)
        key = self.keyL(tokens) # (10, 256) -> (10, 256)
        value = self.valueL(tokens) # (10, 256) -> (10, 256)
        s = torch.matmul(query, torch.t(key)) / self.emb_size[0]**(1/2) # (B, 1, 256) @ (256, 10) -> (B, 1, 10)
        s = torch.squeeze(s, 1)
        attention_weight = self.softmax(s) # (B, 10) -> (B, 10)
        out = torch.matmul(attention_weight, value) #(B, 10) @ (10, 256) -> (B, 256)
        
        return out, attention_weight


class GSTLayer(nn.Module):
    # input : [メルスペクトログラムのlist]
    # out : (B, 256), (B, 10)  
    def __init__(
        self,
        n_mel=80,
    ):
        super().__init__()
        self.ref = ReferenceEncoder(n_mel=n_mel).cuda()
        self.style = StyleTokenLayer().cuda()
        
    def forward(self, seqs=None, token_num=0): #inference時の挙動はreference audioを持ってくる形ではなく、tokenを指定してそれをembeddingとして用いるようにした
        if seqs == None: #inference
            token = self.style(token_num=token_num)
            return token
        else:
            seqs = pad_sequence(seqs, batch_first=True)
            seqs = torch.unsqueeze(seqs, 1)
            _, ref_emb = self.ref(seqs)
            ref_emb = torch.transpose(ref_emb, 0, 1)
            style_out, style_weight = self.style(ref_emb)
            style_out = torch.unsqueeze(style_out, 1)
            return style_out, style_weight