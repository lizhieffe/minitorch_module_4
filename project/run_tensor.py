"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

# Need to access the local package when running in colab
import sys
sys.path.append('/content/minitorch_module_2')

import minitorch
import time



def RParam(*shape):
    # TODO(lizhi): does it need to set requires_grad to True?
    # rand() is defined in tensor_functions.py
    r = 2 * (minitorch.rand(shape) - 0.5)

    # Parameter is defined in module.py
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)

    def forward(self, x):
        y = self.layer1.forward(x)
        y = minitorch.ReLU.apply(y)
        y = self.layer2.forward(y)
        y = minitorch.ReLU.apply(y)
        y = self.layer3.forward(y)
        y = minitorch.Sigmoid.apply(y)
        return y

class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
        self.out_size = out_size

        # print(f"===lizhi run_tensor Linear bias {self.bias.value.unique_id} {self.bias.value.requires_grad()=}")

    def forward(self, x):
        y = minitorch.MatMul.apply(x, self.weights.value)
        y = minitorch.Add.apply(y, self.bias.value)
        return y


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        params = self.model.parameters()
        # print(f"===lizhi run_tensor train params: {[p.value.unique_id for p in params]}")
        optim = minitorch.SGD(params, learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 2
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)


    start_time = time.time()
   
    TensorTrain(HIDDEN).train(data, RATE)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Elapsed time: {elapsed_time} seconds")
