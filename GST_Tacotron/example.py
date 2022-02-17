#トイモデルで動作確認
import torch
from utils.utils import pad_1d
from model.GST_Tacotron import GST_Tacotron


# 語彙の定義
characters = "abcdefghijklmnopqrstuvwxyz!'(),-.:;? "
# その他特殊記号
extra_symbols = [
    "^",  # 文の先頭を表す特殊記号 <SOS>
    "$",  # 文の末尾を表す特殊記号 <EOS>
]
_pad = "~"

# NOTE: パディングを 0 番目に配置
symbols = [_pad] + extra_symbols + list(characters)

# 文字列⇔数値の相互変換のための辞書
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}

def text_to_sequence(text):
    # 簡易のため、大文字と小文字を区別せず、全ての大文字を小文字に変換
    text = text.lower()

    # <SOS>
    seq = [_symbol_to_id["^"]]

    # 本文
    seq += [_symbol_to_id[s] for s in text]

    # <EOS>
    seq.append(_symbol_to_id["$"])

    return seq


def sequence_to_text(seq):
    return [_id_to_symbol[s] for s in seq]

def get_dummy_input():
    # バッチサイズに 2 を想定して、適当な文字列を作成
    seqs = [
        text_to_sequence("What is your favorite language?"),
        text_to_sequence("Hello world."),
    ]
    in_lens = torch.tensor([len(x) for x in seqs], dtype=torch.long)
    max_len = max(len(x) for x in seqs)
    seqs = torch.stack([torch.from_numpy(pad_1d(seq, max_len)) for seq in seqs])
    
    return seqs, in_lens

def get_dummy_inout():
    seqs, in_lens = get_dummy_input()
   
    # デコーダの出力（メルスペクトログラム）の教師データ
    decoder_targets = [torch.ones(120, 80).to("cuda"), torch.ones(120, 80).to("cuda")]
    
    # stop token の教師データ
    # stop token の予測値は確率ですが、教師データは 二値のラベルです
    # 1 は、デコーダの出力が完了したことを表します
    stop_tokens = torch.zeros(2, 120)
    stop_tokens[:, -1:] = 1.0
    
    return seqs, in_lens, decoder_targets, stop_tokens

seqs, in_lens, decoder_targets, stop_tokens = get_dummy_inout()
model = GST_Tacotron()
model.cuda()
outs, outs_fine, encoder_outs, logits, att_ws, gst_att_ws = model(seqs.to("cuda"), in_lens, decoder_targets)

print("入力のサイズ:", tuple(seqs.shape))
print("エンコーダの出力のサイズ:", tuple(encoder_outs.shape))
print("GSTのアテンション重みのサイズ:", tuple(gst_att_ws.shape))
print("デコーダの出力のサイズ:", tuple(outs.shape))
print("Stop token のサイズ:", tuple(logits.shape))
print("デコーダのアテンション重みのサイズ:", tuple(att_ws.shape))