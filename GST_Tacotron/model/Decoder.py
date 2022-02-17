import torch
from torch import nn
from torch.nn import functional as F
from utils.utils import make_pad_mask

class LocationSensitiveAttention(nn.Module):
    def __init__(
        self,
        encoder_dim=512,
        decoder_dim=1024,
        hidden_dim=128,
        conv_channels=32,
        conv_kernel_size=31,
    ):
        super().__init__()
        self.V = nn.Linear(encoder_dim, hidden_dim)
        self.W = nn.Linear(decoder_dim, hidden_dim, bias=False)
        self.U = nn.Linear(conv_channels, hidden_dim, bias=False)
        self.F = nn.Conv1d(
            1,
            conv_channels,
            conv_kernel_size,
            padding=(conv_kernel_size - 1) // 2,
            bias=False,
        )
        # NOTE: 本書の数式通りに実装するなら bias=False ですが、実用上は bias=True としても問題ありません
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, encoder_outs, src_lens, decoder_state, att_prev, mask=None):
        # アテンション重みを一様分布で初期化
        if att_prev is None:
            att_prev = 1.0 - make_pad_mask(src_lens).to(
                device=decoder_state.device, dtype=decoder_state.dtype
            )
            att_prev = att_prev / src_lens.unsqueeze(-1).to(encoder_outs.device)

        # (B x T_enc) -> (B x 1 x T_enc) -> (B x conv_channels x T_enc) ->
        # (B x T_enc x conv_channels)
        f = self.F(att_prev.unsqueeze(1)).transpose(1, 2)

        # 式 (9.13) の計算
        erg = self.w(
            torch.tanh(
                self.W(decoder_state).unsqueeze(1) + self.V(encoder_outs) + self.U(f)
            )
        ).squeeze(-1)

        if mask is not None:
            erg.masked_fill_(mask, -float("inf"))

        attention_weights = F.softmax(erg, dim=1)

        # エンコーダ出力の長さ方向に対して重み付き和を取ります
        attention_context = torch.sum(
            encoder_outs * attention_weights.unsqueeze(-1), dim=1
        )

        return attention_context, attention_weights

class Prenet(nn.Module):
    def __init__(self, in_dim, layers=2, hidden_dim=256, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        prenet = nn.ModuleList()
        for layer in range(layers):
            prenet += [
                nn.Linear(in_dim if layer == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
            ]
        self.prenet = nn.Sequential(*prenet)

    def forward(self, x):
        for layer in self.prenet:
            # 学習時、推論時の両方で Dropout を適用します
            x = F.dropout(layer(x), self.dropout, training=True)
        return x

class ZoneOutCell(nn.Module):
    def __init__(self, cell, zoneout=0.1):
        super().__init__()
        self.cell = cell
        self.hidden_size = cell.hidden_size
        self.zoneout = zoneout

    def forward(self, inputs, hidden):
        next_hidden = self.cell(inputs, hidden)
        next_hidden = self._zoneout(hidden, next_hidden, self.zoneout)
        return next_hidden

    def _zoneout(self, h, next_h, prob):
        h_0, c_0 = h
        h_1, c_1 = next_h
        h_1 = self._apply_zoneout(h_0, h_1, prob)
        c_1 = self._apply_zoneout(c_0, c_1, prob)
        return h_1, c_1

    def _apply_zoneout(self, h, next_h, prob):
        if self.training:
            mask = h.new(*h.size()).bernoulli_(prob)
            return mask * h + (1 - mask) * next_h
        else:
            return prob * h + (1 - prob) * next_h

class Decoder(nn.Module):
    def __init__(
        self,
        encoder_hidden_dim=256,
        out_dim=80,
        layers=2,
        hidden_dim=1024,
        prenet_layers=2,
        prenet_hidden_dim=256,
        prenet_dropout=0.5,
        zoneout=0.1,
        reduction_factor=1,
        attention_hidden_dim=128,
        attention_conv_channels=32,
        attention_conv_kernel_size=31,
    ):
        super().__init__()
        self.out_dim = out_dim

        # 注意機構
        self.attention = LocationSensitiveAttention(
            encoder_hidden_dim,
            hidden_dim,
            attention_hidden_dim,
            attention_conv_channels,
            attention_conv_kernel_size,
        )
        self.reduction_factor = reduction_factor

        # Prenet
        self.prenet = Prenet(out_dim, prenet_layers, prenet_hidden_dim, prenet_dropout)

        # 片方向LSTM
        self.lstm = nn.ModuleList()
        for layer in range(layers):
            lstm = nn.LSTMCell(
                encoder_hidden_dim + prenet_hidden_dim if layer == 0 else hidden_dim,
                hidden_dim,
            )
            lstm = ZoneOutCell(lstm, zoneout)
            self.lstm += [lstm]

        # 出力への projection 層
        proj_in_dim = encoder_hidden_dim + hidden_dim
        self.feat_out = nn.Linear(proj_in_dim, out_dim * reduction_factor, bias=False)
        self.prob_out = nn.Linear(proj_in_dim, reduction_factor)

    def _zero_state(self, hs):
        init_hs = hs.new_zeros(hs.size(0), self.lstm[0].hidden_size)
        return init_hs

    def forward(self, encoder_outs, in_lens, decoder_targets=None):
        is_inference = decoder_targets is None

        # Reduction factor に基づくフレーム数の調整
        # (B, Lmax, out_dim) ->  (B, Lmax/r, out_dim)
        if self.reduction_factor > 1 and not is_inference:
            decoder_targets = decoder_targets[
                :, self.reduction_factor - 1 :: self.reduction_factor
            ]

        # デコーダの系列長を保持
        # 推論時は、エンコーダの系列長から経験的に上限を定める
        if is_inference:
            max_decoder_time_steps = int(encoder_outs.shape[1] * 10.0)
        else:
            max_decoder_time_steps = decoder_targets.shape[1]

        # ゼロパディングされた部分に対するマスク
        mask = make_pad_mask(in_lens).to(encoder_outs.device)

        # LSTM の状態をゼロで初期化
        h_list, c_list = [], []
        for _ in range(len(self.lstm)):
            h_list.append(self._zero_state(encoder_outs))
            c_list.append(self._zero_state(encoder_outs))

        # デコーダの最初の入力
        go_frame = encoder_outs.new_zeros(encoder_outs.size(0), self.out_dim)
        prev_out = go_frame

        # 1つ前の時刻のアテンション重み
        prev_att_w = None

        # メインループ
        outs, logits, att_ws = [], [], []
        t = 0
        while True:
            # コンテキストベクトル、アテンション重みの計算
            att_c, att_w = self.attention(
                encoder_outs, in_lens, h_list[0], prev_att_w, mask
            )

            # Pre-Net
            prenet_out = self.prenet(prev_out)

            # LSTM
            xs = torch.cat([att_c, prenet_out], dim=1)
            h_list[0], c_list[0] = self.lstm[0](xs, (h_list[0], c_list[0]))
            for i in range(1, len(self.lstm)):
                h_list[i], c_list[i] = self.lstm[i](
                    h_list[i - 1], (h_list[i], c_list[i])
                )
            # 出力の計算
            hcs = torch.cat([h_list[-1], att_c], dim=1)
            outs.append(self.feat_out(hcs).view(encoder_outs.size(0), self.out_dim, -1))
            logits.append(self.prob_out(hcs))
            att_ws.append(att_w)

            # 次の時刻のデコーダの入力を更新
            if is_inference:
                prev_out = outs[-1][:, :, -1]  # (1, out_dim)
            else:
                # Teacher forcing
                prev_out = decoder_targets[:, t, :]

            # 累積アテンション重み
            prev_att_w = att_w if prev_att_w is None else prev_att_w + att_w

            t += 1
            # 停止条件のチェック
            if t >= max_decoder_time_steps:
                break
            if is_inference and (torch.sigmoid(logits[-1]) >= 0.5).any():
                break
                
        # 各時刻の出力を結合
        logits = torch.cat(logits, dim=1)  # (B, Lmax)
        outs = torch.cat(outs, dim=2)  # (B, out_dim, Lmax)
        att_ws = torch.stack(att_ws, dim=1)  # (B, Lmax, Tmax)

        if self.reduction_factor > 1:
            outs = outs.view(outs.size(0), self.out_dim, -1)  # (B, out_dim, Lmax)

        return outs, logits, att_ws