import torch

class SingleEncoder(torch.nn.Module):
    """defines task encoder based on single-step transitions"""
    def __init__(self,ns,na,latent_dim,cov_type='diag',hs=500):
        super(SingleEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(2*ns+na+1, hs)
        self.fc2 = torch.nn.Linear(hs, hs)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hs)
        self.ln = torch.nn.LayerNorm(hs)
        if cov_type=='diag': #one std-dev parameter for each dim
            self.fc3 = torch.nn.Linear(hs, 2*latent_dim)
        elif cov_type=='scalar': #uniform standard deviation
            self.fc3 = torch.nn.Linear(hs, latent_dim+1)
        elif cov_type=='dense': #full, dense covariance matrix
            self.fc3 = torch.nn.Linear(hs, latent_dim+(latent_dim*((latent_dim+1))/2)) #need latent + latent'th triangle num.
        elif cov_type=='fixed': #use standard covariance
            self.fc3 = torch.nn.Linear(hs,latent_dim)
        else:
            raise ValueException('invalid covariance type')
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class TransitionNet(torch.nn.Module):
    def __init__(self,ns,na,latent_dim,cov_type='diag',hs=500):
        super(TransitionNet, self).__init__()
        self.fc1 = torch.nn.Linear(ns+na+latent_dim, hs)
        self.fc2 = torch.nn.Linear(hs, hs)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hs)
        self.ln = torch.nn.LayerNorm(hs)
        if cov_type=='diag': #one std-dev parameter for each dim
            self.fc3 = torch.nn.Linear(hs, 2*ns)
        elif cov_type=='scalar': #uniform standard deviation
            self.fc3 = torch.nn.Linear(hs, ns+1)
        elif cov_type=='dense': #full, dense covariance matrix
            self.fc3 = torch.nn.Linear(hs, ns+(ns*((ns+1))/2)) #need num_state + num_state'th triangle num.
        elif cov_type=='fixed': #use standard covariance
            self.fc3 = torch.nn.Linear(hs,ns)
        else:
            raise ValueException('invalid covariance type')
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

class RewardNet(torch.nn.Module):
    def __init__(self,ns,na,latent_dim,hs=100):
        super(RewardNet, self).__init__()
        self.fc1 = torch.nn.Linear(ns+na+latent_dim, hs)
        self.fc2 = torch.nn.Linear(hs, hs)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hs)
        self.ln = torch.nn.LayerNorm(hs)
        self.fc3 = torch.nn.Linear(hs,2)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    
class LSTMEncoder(torch.nn.Module):
    def __init__(self,ns,na,latent_dim,hs=200,num_layers=2,cov_type='diag',batch_first=True):
        super(LSTMEncoder, self).__init__()
        self.layers = num_layers
        self.hs = hs
        self.latent_dim = latent_dim
        self.LSTM = torch.nn.LSTM(2*ns+na+1,hs,num_layers,batch_first=batch_first)
        self.relu = torch.nn.ReLU()
        if cov_type=='diag': #one std-dev parameter for each dim
            self.linear = torch.nn.Linear(hs, 2*latent_dim)
        elif cov_type=='scalar': #uniform standard deviation
            self.linear = torch.nn.Linear(hs, latent_dim+1)
        elif cov_type=='dense': #full, dense covariance matrix
            self.linear = torch.nn.Linear(hs, latent_dim+(latent_dim*((latent_dim+1))/2)) #need latent + latent'th triangle num.
        elif cov_type=='fixed': #use standard covariance
            self.linear = torch.nn.Linear(hs,latent_dim)
        else:
            raise ValueException('invalid covariance type')
            
    def forward(self, x, hidden=None):
        if hidden is None:
            batch_size = x.shape[0]
            hidden = self.init_hidden(batch_size)
        x, hidden = self.LSTM(x,hidden)
        x = self.linear(self.relu(x))
        return x, hidden
    
    def init_hidden(self,batch_size):
        h0 = torch.zeros(self.layers,batch_size,self.hs)
        c0 = torch.zeros(self.layers,batch_size,self.hs)
        return (h0,c0)