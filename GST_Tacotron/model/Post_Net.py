from torch import nn

class Postnet(nn.Module):
    def __init__(
        self,
        in_dim=80,
        layers=5,
        channels=512,
        kernel_size=5,
        dropout=0.5,
    ):
        super().__init__()
        postnet = nn.ModuleList()
        for layer in range(layers):
            in_channels = in_dim if layer == 0 else channels
            out_channels = in_dim if layer == layers - 1 else channels
            postnet += [
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            ]
            if layer != layers - 1:
                postnet += [nn.Tanh()]
            postnet += [nn.Dropout(dropout)]
        self.postnet = nn.Sequential(*postnet)

    def forward(self, xs):
        return self.postnet(xs)