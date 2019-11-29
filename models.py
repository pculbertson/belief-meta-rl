import torch

class TransitionNet(torch.nn.Module):
    def __init__(self,ns,na,cov_type='diag',hs=500):
        super(TransitionNet, self).__init__()
        self.fc1 = torch.nn.Linear(ns+na, hs)
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
    def __init__(self,ns,na,cov=True,hs=100):
        super(RewardNet, self).__init__()
        self.fc1 = torch.nn.Linear(ns+na, hs)
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