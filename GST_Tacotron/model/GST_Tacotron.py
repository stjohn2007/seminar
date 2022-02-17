import torch
from torch import nn

from GST import GSTLayer
from Encoder import Encoder
from Decoder import Decoder
from Post_Net import Postnet

class GST_Tacotron(nn.Module):
    def __init__(self
    ):
        super().__init__()
        self.encoder = Encoder()
        self.gst = GSTLayer()
        self.decoder = Decoder()
        self.postnet = Postnet()

    def forward(self, seq, in_lens, decoder_targets):
        # エンコーダによるテキストに潜在する表現の獲得
        encoder_outs = self.encoder(seq, in_lens)
        
        # GSTによる音声に潜在する表現の獲得
        gst_outs, gst_att_ws = self.gst(decoder_targets)
        
        # エンコーダの出力とGSTの出力を足す
        encoder_outs += gst_outs.expand_as(encoder_outs) 
        print(encoder_outs.size())

        # デコーダによるメルスペクトログラム、stop token の予測
        outs, logits, att_ws = self.decoder(encoder_outs, in_lens, torch.stack(decoder_targets))

        # Post-Net によるメルスペクトログラムの残差の予測
        outs_fine = outs + self.postnet(outs)

        # (B, C, T) -> (B, T, C)
        outs = outs.transpose(2, 1)
        outs_fine = outs_fine.transpose(2, 1)

        return outs, outs_fine, encoder_outs, logits, att_ws, gst_att_ws 
    
    def inference(self, seq):
        seq = seq.unsqueeze(0) if len(seq.shape) == 1 else seq
        in_lens = torch.tensor([seq.shape[-1]], dtype=torch.long, device=seq.device)

        return self.forward(seq, in_lens, None)