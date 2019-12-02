import torch

class SingleEncoder(torch.nn.Module):
    """defines task encoder based on single-step transitions"""
    def __init__(self,ns,na,latent_dim,cov_type='diag',hs=500):
        super(TransitionNet, self).__init__()
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
    def __init__(self,ns,na,latent_dim,cov=True,hs=100):
        super(RewardNet, self).__init__()
        self.fc1 = torch.nn.Linear(ns+na+latent_dim, hs)
        self.fc2 = torch.nn.Linear(hs, hs)
        self.relu = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm1d(hs)
        self.ln = torch.nn.LayerNorm(hs)
        if cov:
            self.fc3 = torch.nn.Linear(hs,2)
        else:
            self.fc3 = torch.nn.Linear(hs,1)
            
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.ln(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x