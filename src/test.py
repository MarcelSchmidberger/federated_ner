import torch
import syft as sy
import torch.nn.functional as F 
import torch.nn as nn

print(f"Syft Version: {sy.version.__version__}")
print(f"Torch Version: {torch.__version__}")

hook = sy.TorchHook(torch)
hook.local_worker.is_client_worker = False

class Net(sy.Plan):
    def __init__(self):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(10, 2)
        self.fc1 = nn.Linear(2, 3)
        self.fc2 = nn.Linear(3, 2)

    def forward(self, x):
        x = self.embedding(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=0)

model = Net()
model.build(torch.tensor([1., 2.]).long())
