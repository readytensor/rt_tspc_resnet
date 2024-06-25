# This is a wrapper class to train and predict deep learning models using torch.
import math
import os
from typing import Callable, Union
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn import Flatten, Softmax, Linear, Module, CrossEntropyLoss, Dropout, ReLU
import torch.optim as optim
from tqdm import tqdm


MODEL_PARAMS_FNAME = "model_params.save"
MODEL_WTS_FNAME = "model_wts.save"



class ResidualBlock(Module):
    def __init__(self, input_size, hidden_size):
        super(ResidualBlock, self).__init__()
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, input_size)
        self.activation = ReLU()
        self.dropout = Dropout(p=0.1)

    def forward(self, x):
        identity = x  # Save the input for the skip connection
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out += identity  # Add the input (skip connection)
        out = self.activation(out)
        return out


class Net(Module):
    def __init__(self, feat_dim, encode_len, n_classes, activation, lr):
        """
        This is a wrapper class for the nerual net model.

        Args:

            feat_dim (int): Number of features in the input data.
            encode_len (int): Encoding (history) length.
            n_classes (int): Number of classes to predict.
            activation (str): Activation function to use in the model.
            lr (float): Learning rate for the model.
        """
        super(Net, self).__init__()
        self.MODEL_NAME = "ResNet_Timeseries_Annotator_residual_blocks"

        self.feat_dim = feat_dim
        self.encode_len = encode_len
        self.n_classes = n_classes
        self.lr = lr
        self.activation = get_activation(activation)
        self.device = get_device()
        self.print_period = 1

        self.softmax = Softmax(dim=-1)
        self.criterion = CrossEntropyLoss()
        self.dropout = Dropout(p=0.1)

        # Define dimensions
        dim1 = 1000
        dim2 = 200
        dim3 = 100
        dim4 = 100

        # Initial fully connected layer
        self.fc1 = Linear(self.feat_dim, dim1)

        # Residual blocks
        self.res_block1 = ResidualBlock(dim1, dim2)
        self.res_block2 = ResidualBlock(dim1, dim2)

        # Final fully connected layers after residual blocks
        self.fc2 = Linear(dim1, dim3)
        self.fc3 = Linear(dim3, dim4)

        self.output = Linear(dim4 * self.encode_len,
                             self.encode_len * self.n_classes)
        self.flatten = Flatten()

    def forward(self, X):
        batch_size = X.size(0)
        x = X.permute(0, 1, 2)  # keeping it as is

        x = self.fc1(x)
        x = self.activation(x)

        x = self.res_block1(x)
        x = self.res_block2(x)

        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)

        x = x.reshape(batch_size, -1)
        x = self.output(x)

        # Reshape to (batch_size, encode_len, n_classes)
        x = x.view(-1, self.encode_len, self.n_classes)

        return x

    def fit(self, train_X, train_y, valid_X=None, valid_y=None, max_epochs=100, batch_size=64, verbose=1):

        patience = get_patience_factor(train_X.shape[0])

        train_X, train_y = torch.FloatTensor(
            train_X), torch.FloatTensor(train_y)
        train_dataset = CustomDataset(train_X, train_y)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=int(batch_size),
            shuffle=True
        )

        if valid_X is not None and valid_y is not None:
            valid_X, valid_y = torch.FloatTensor(
                valid_X), torch.FloatTensor(valid_y)
            valid_dataset = CustomDataset(valid_X, valid_y)
            valid_loader = DataLoader(
                dataset=valid_dataset, batch_size=int(batch_size), shuffle=True
            )
        else:
            valid_loader = None

        losses = self._run_training(
            train_loader,
            valid_loader,
            max_epochs,
            use_early_stopping=True,
            patience=patience,
            verbose=verbose,
        )
        return losses

    def predict_proba(self, x):
        x = torch.FloatTensor(x).to(self.device)
        x = self.forward(x)
        x = self.softmax(x)
        return x.cpu().detach().numpy()

    def predict(self, x):
        x = torch.FloatTensor(x).to(self.device)
        x = self.forward(x)
        x = self.softmax(x)
        x = torch.argmax(x, dim=-1)
        return x.cpu().detach().numpy()

    def _run_training(
        self,
        train_loader,
        valid_loader,
        max_epochs,
        use_early_stopping=True,
        patience=10,
        verbose=1,
    ):

        best_loss = 1e7
        losses = []
        min_epochs = 10
        for epoch in range(max_epochs):
            self.train()
            for data in tqdm(train_loader, total=len(train_loader)):
                X, y = data[0].to(self.device), data[1].to(self.device)
                preds = self(X)
                loss = self.criterion(
                    preds.view(-1, preds.size(-1)), y.view(-1).long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            current_loss = loss.item()

            if use_early_stopping:
                if valid_loader is not None:
                    current_loss = self.get_loss(
                        valid_loader, self.criterion
                    )
                losses.append({"epoch": epoch, "loss": current_loss})
                if current_loss < best_loss:
                    trigger_times = 0
                    best_loss = current_loss
                else:
                    trigger_times += 1
                    if trigger_times >= patience and epoch >= min_epochs:
                        if verbose == 1:
                            print(f"Early stopping after {epoch=}!")
                        return losses

            else:
                losses.append({"epoch": epoch, "loss": current_loss})
            # Show progress
            if verbose == 1:
                if epoch % self.print_period == 0 or epoch == max_epochs - 1:
                    print(
                        f"Epoch: {epoch+1}/{max_epochs}, loss: {np.round(current_loss, 5)}"
                    )

        return losses

    def save(self, model_path):
        model_params = {
            "encode_len": self.encode_len,
            "feat_dim": self.feat_dim,
            "activation": self.activation,
            "n_classes": self.n_classes,
            "lr": self.lr,
        }
        joblib.dump(model_params, os.path.join(model_path, MODEL_PARAMS_FNAME))
        torch.save(self.state_dict(), os.path.join(
            model_path, MODEL_WTS_FNAME))

    @classmethod
    def load(cls, model_path):
        model_params = joblib.load(
            os.path.join(model_path, MODEL_PARAMS_FNAME))
        model = cls(**model_params)
        model.load_state_dict(
            torch.load(os.path.join(model_path, MODEL_WTS_FNAME))
        )
        return model

    def __str__(self):
        return f"Model name: {self.MODEL_NAME}"

    def set_optimizer(self, optimizer_name):
        if optimizer_name == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        else:
            raise ValueError(
                f"Error: Unrecognized optimizer type: {optimizer_name}. "
                "Must be one of ['adam', 'sgd']."
            )

    def get_num_parameters(self):
        pp = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp += nn
        return pp

    def get_loss(self, data_loader, loss_function):
        self.eval()
        loss_total = 0
        with torch.no_grad():
            for data in data_loader:
                X, y = data[0].to(self.device), data[1].to(self.device)
                output = self(X)
                loss = loss_function(y, output)
                loss_total += loss.item()
        return loss_total / len(data_loader)


def get_device():
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    return device


def get_activation(activation) -> Callable:
    """
    Return the activation function based on the input string.

    This function returns a callable activation function from the
    torch.nn.functional package.

    Args:
        activation: Name of the activation function or the function itself.

    Returns:
        Callable: The requested activation function. If 'none' is specified,
        it will return an identity function.

    Raises:
        Exception: If the activation string does not match any known
        activation functions ('relu', 'tanh', or 'none').

    """
    if activation == "tanh" or activation == F.tanh:
        return F.tanh
    elif activation == "relu" or activation == F.relu:
        return F.relu
    elif activation == "none":
        return lambda x: x  # Identity function, doesn't change input
    else:
        raise ValueError(
            f"Error: Unrecognized activation type: {activation}. "
            "Must be one of ['relu', 'tanh', 'none']."
        )


def get_patience_factor(N):
    # magic number - just picked through trial and error
    if N < 100:
        return 30
    patience = int(37 - math.log(N, 1.5))
    return patience


class CustomDataset(Dataset):

    def __init__(self, x, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        if self.y is None:
            return self.x[index]
        else:
            return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print("device used: ", device)

    N = 1
    T = 20
    D = 2
    encode_len = T

    model = Net(
        feat_dim=D,
        encode_len=encode_len,
        n_classes=10,
        activation="relu",
        lr=0.001,
    )
    model.to(device=device)

    X = np.random.randn(
        N, encode_len, D).astype(np.float32)

    print(model)

    preds = model.predict_proba(X)
    print(preds)
    print("output", preds.shape)
