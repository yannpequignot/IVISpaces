import torch
from torch import nn
import math


class IVI():
    def __init__(self, train_loader, ELBO,
                 optimizer):
        
        self.train_loader=train_loader
        self.ELBO=ELBO
        
        self.optimizer=optimizer



    def run(self, GeN, C):

        self.scores={'ELBO':0. ,
                     'KL':0.,
                     'LOSS':0.
        }
        example_count=0.
        
        GeN.train(True)
        with torch.enable_grad():
            for (x,y) in self.train_loader:

                self.optimizer.zero_grad()
                
                
                L, K, LL=self.ELBO(x,y,GeN, C)
                L.backward()

                lr = self.optimizer.param_groups[0]['lr']

                self.optimizer.step()

                self.scores['ELBO']+= L.item()*len(x)
                self.scores['KL']+= K.item()*len(x)
                self.scores['LOSS']+=LL.item()*len(x)
                example_count+=len(x)
    
        mean_scores={'ELBO': self.scores['ELBO']/example_count ,
             'KL':self.scores['KL']/example_count,
             'LOSS':self.scores['LOSS']/example_count,
             'lr':lr
            }
        return mean_scores
    
