import torch
from torch import nn
import math

from tqdm import trange

from Metrics import KL, batchKL



class FuNNeVI():
    def __init__(self, loglikelihood, batch, size_data, prior, projection, p, kNNE,
                 n_samples_KL, n_samples_LL, max_iter, learning_rate, min_lr, patience, lr_decay, device):
        
        self.loglikelihood=loglikelihood
        self.batch=batch
        self.size_data=size_data
        self.prior=prior
        self.projection=projection
        self.rho=batch/size_data
        
        self.p=p     
        self.kNNE=kNNE
        self.n_samples_KL=n_samples_KL
        self.n_samples_LL=n_samples_LL
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.patience = patience
        self.lr_decay = lr_decay
        self.device = device


    def _KL(self,GeN):

        theta=GeN(self.n_samples_KL) #variationnel
        theta_prior=self.prior(self.n_samples_KL) #prior

        theta_proj, theta_prior_proj = self.projection(theta, theta_prior)

        K=KL(theta_proj, theta_prior_proj,k=self.kNNE,device=self.device,p=self.p)
        return K

    def run(self, GeN, show_fn=None):
        one_epoch=int(self.size_data/self.batch)

        optimizer = torch.optim.Adam(GeN.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.patience, factor=self.lr_decay)

        self.scores={'ELBO': [] ,
                     'KL':[],
                     'LL':[],
                     'lr':[]
        }

        with trange(self.max_iter) as tr:
            for t in tr:

                optimizer.zero_grad()

                K = self._KL(GeN) #KL(Q_var,Prior)
                LL = self.loglikelihood(GeN(self.n_samples_LL), self.batch).mean()
                L=self.rho*K-LL
                L.backward()

                lr = optimizer.param_groups[0]['lr']


                tr.set_postfix(ELBO=L.item(), LogLike=LL.item(), KL=K.item(), lr=lr)

                optimizer.step()


                if t % 100 ==0:
                    self.scores['ELBO'].append(L.item())
                    self.scores['KL'].append(K.item())
                    self.scores['LL'].append(LL.item())
                    self.scores['lr'].append(lr)

                if lr < self.min_lr:
                      break
                scheduler.step(L.item())
        return L