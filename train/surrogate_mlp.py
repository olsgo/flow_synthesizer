import torch, torch.nn as nn

class SurrogateMLP(nn.Module):
    def __init__(self, in_dim=66, out_dim=483, hidden=(256,256,256), p=0.1):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.GELU(), nn.Dropout(p)]
            last = h
        layers += [nn.Linear(last, out_dim)]  # features are real-valued (no sigmoid)
        self.net = nn.Sequential(*layers)

    def forward(self, params66, probe_onehot=None):
        x = params66
        if probe_onehot is not None:
            x = torch.cat([params66, probe_onehot], dim=-1)
        return self.net(x)
