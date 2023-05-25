import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['KaiTi']
input_size = 7
hidden_size = 7
batch_size = 1
num_layers=1

idx2char = ['h', 'i', 'n', 'r', 's', 't', ' ']
x_data =[2, 3, 3, 6, 1, 4, 6, 1, 4, 5, 0]  # nrr is isth
y_data =[5, 0, 1, 4, 6, 1, 4, 6, 3, 2, 2]  # this is rnn

one_hot_lookup = [[1, 0, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0, 0],
                  [0, 0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1]]
x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
labels = torch.LongTensor(y_data)

class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, num_layers):
        super(Model, self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = torch.nn.RNN(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=num_layers)

    def forward(self, input):
        hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        out, _ = self.rnn(input, hidden)
        return out.view(-1, self.hidden_size)

net = Model(input_size, hidden_size, batch_size, num_layers)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted:', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/100] loss=%.3f' % (epoch+1, loss.item()))
