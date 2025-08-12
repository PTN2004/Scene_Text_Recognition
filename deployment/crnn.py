import torch.nn as nn
import timm


class CRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layer, unfreeze_layers=3, dropout=0.2):
        super(CRNN, self).__init__()
        backbone = timm.create_model("resnet152", pretrained=True, in_chans=1)
        # Tuong tac voi resnet152 model
        modules = list(backbone.children())[:-2]
        modules.append(nn.AdaptiveAvgPool2d((1, None)))
        self.backbone = nn.Sequential(*modules)

        for para in self.backbone[-unfreeze_layers:].parameters():
            para.requires_grad = True

        self.mapSeq = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.GRU = nn.GRU(
            512, hidden_size,
            num_layer,
            dropout=dropout if num_layer > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(hidden_size*2)

        self.out = nn.Sequential(
            nn.Linear(hidden_size*2, vocab_size),
            nn.LogSoftmax(2)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(x.size(0), x.size(1), -1)
        x = self.mapSeq(x)
        x, _ = self.GRU(x)
        x = self.layer_norm(x)
        x = self.out(x)
        x = x.permute(1, 0, 2)  # based CTC loss
        return x

