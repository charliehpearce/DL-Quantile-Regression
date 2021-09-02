import torch as t 
import numpy as np
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch import tensor


class QuantReg(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super(QuantReg, self).__init__()
        
        # Define basic LSTM neural net
        #self.LSTM = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,\
        #    batch_first = True)
        self.net = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_size, 2))
    
    def forward(self, X):
        out = self.net(X)
        return out

class DLUtils:
    @staticmethod
    def loss_fn(y_hat, y, p_val):
        """
        Args:
        y_hat - tensor of shape (n,2) of two intervals
        y - true value from regression
        p_val - target p_val of model
        """
        def get_error(y_hat_q, y, q):
            e = y - y_hat_q
            return t.mean(t.max(q*e, e*(q-1)),dim=-1)

        # Calculate quantiles from p_value
        quantile1 = p_val/2
        quantile2 = 1-(p_val/2)

        # calculate errors for both
        error1 = get_error(y_hat[:,0],y,quantile1)
        error2 = get_error(y_hat[:,1],y,quantile2)
        
        return error1+error2

    def train(self, model, n_epoch, optim, learning_rate, train_loader, p_val):
        optimizer = optim(model.parameters(),learning_rate)
        #loss_fn = nn.MSELoss()
        
        for epoch in range(n_epoch):
            model.train()
            losses = []
            for feats, labs in train_loader:
                optimizer.zero_grad()
                out = model(feats)
                l = self.loss_fn(out, labs, p_val)
                losses.append(l)
                l.backward()
                optimizer.step()
        
            print(f'Epoch {epoch+1} loss : {t.mean(l).detach()}')
        return model

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing as data
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    #np.random.seed(1234)

    scaler = MinMaxScaler()
    X, y = data(return_X_y = True)
    X_scaled = scaler.fit_transform(X)

    rmses=[]
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=1234)
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    train_tensor = TensorDataset(t.tensor(X_train),t.tensor(y_train))
    train_loader = DataLoader(train_tensor, batch_size=100, shuffle=True)

    train_tensor = TensorDataset(t.tensor(X_train),t.tensor(y_train))
    train_loader = DataLoader(train_tensor, batch_size=100, shuffle=True)

    print('defining model')
    model = QuantReg(input_size=8, hidden_size=100, num_layers=2)

    print('training model')
    trained_model = DLUtils().train(model=model, n_epoch=50, optim=t.optim.Adam, learning_rate=0.01, train_loader=train_loader, p_val=0.05)

    print('model eval')
    trained_model.eval()
    y_pred = trained_model(tensor(X_test)).detach().numpy()