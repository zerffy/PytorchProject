from torch import nn

class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.conv = nn.Sequential(
            OrderedDict(
                [
                ("conv1", nn.Conv2d(3, 32, 3, 1, 1)),
                ("relu1", nn.ReLU()),
                ("pool", nn.MaxPool2d(2))
                ]
            ))
        self.dense = nn.Sequential(
            OrderedDict([
                ("dense1", nn.Linear(32 * 3 * 3, 128)),
                ("relu2", nn.ReLU()),
                ("dense2", nn.Linear(128, 10))
                ])
        )
