import torch, torch.nn as nn

class PredictorMLP(nn.Module):
    def __init__(self, in_dim, out_params=66, hidden=(1024,512,256), p=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(p)]
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head_params = nn.Sequential(nn.Linear(last, out_params), nn.Sigmoid())
        # Optional: add discrete heads, e.g., waveform choices:
        self.discrete_heads = nn.ModuleDict({})  # name -> Linear(num_classes)

    def forward(self, x):
        h = self.backbone(x)
        y = self.head_params(h)
        logits = {k: head(h) for k,head in self.discrete_heads.items()}
        return y, logits
