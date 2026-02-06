import time, torch, torch.nn as nn, torch.nn.functional as F
from .device import get_device, device_report

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

def run_demo(steps=20, batch_size=256):
    device = get_device()
    print(device_report(device))
    model = TinyNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    t0 = time.time()
    for s in range(steps):
        x = torch.randn(batch_size, 16, device=device)
        y = torch.randint(0, 4, (batch_size,), device=device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    print("done in", round(time.time()-t0,2), "sec")
