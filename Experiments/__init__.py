from Metrics import RMSE, LPP, PICP, MPIW

import tempfile
import mlflow
import matplotlib.pyplot as plt
import importlib.util
import torch

from Tools import logmvn01pdf, NormalLogLikelihood

from sklearn.model_selection import train_test_split
from Preprocessing import fitStandardScalerNormalization, normalize

test_ratio=0.1

def switch_setup(setup):
    return {
        'foong2d':  importlib.util.spec_from_file_location("foong", "Experiments/foong2D/__init__.py") ,
        'foong':  importlib.util.spec_from_file_location("foong", "Experiments/foong/__init__.py") ,
        'foong_sparse':  importlib.util.spec_from_file_location("foong_sparse", "Experiments/foong_sparse/__init__.py") ,
        'foong_mixed':  importlib.util.spec_from_file_location("foong_mixed", "Experiments/foong_mixed/__init__.py") ,
        'boston': importlib.util.spec_from_file_location("boston", "Experiments/boston/__init__.py"),
        'california': importlib.util.spec_from_file_location("california", "Experiments/california/__init__.py"),
        'concrete': importlib.util.spec_from_file_location("concrete", "Experiments/concrete/__init__.py"),
        'energy':  importlib.util.spec_from_file_location("energy", "Experiments/energy/__init__.py") ,
        'wine': importlib.util.spec_from_file_location("wine", "Experiments/winequality/__init__.py"),
        'kin8nm': importlib.util.spec_from_file_location("kin8nm", "Experiments/kin8nm/__init__.py"),
        'powerplant': importlib.util.spec_from_file_location("powerplant", "Experiments/ccpowerplant/__init__.py"),
        'yacht': importlib.util.spec_from_file_location("yacht", "Experiments/yacht/__init__.py"),
        'navalC': importlib.util.spec_from_file_location("navalC", "Experiments/naval/__init__.py"),
        'protein': importlib.util.spec_from_file_location("protein", "Experiments/protein/__init__.py")

    }[setup]


def get_setup(setup):
    spec=switch_setup(setup)
    setup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup)
    return setup


def log_exp_metrics(evaluate_metrics, theta_ens, execution_time, device):
    mlflow.set_tag('execution_time ', '{0:.2f}'.format(execution_time)+'s')

    LPP_test, RMSE_test, RMSE_train, PICP_test, MPIW_test = evaluate_metrics(theta_ens, device)

    mlflow.log_metric("LPP_test", LPP_test[0].item())

    mlflow.log_metric("LPP_test_std", LPP_test[1].item())

    mlflow.log_metric("RMSE_test", RMSE_test[0].item())
    mlflow.log_metric("RMSE_train", RMSE_train[0].item())

    mlflow.log_metric("RSSE_test_std", RMSE_test[1].item())
    
    mlflow.log_metric("PICP_test", PICP_test.item())
    mlflow.log_metric("MPIW_test", MPIW_test.item())

    return LPP_test, RMSE_test, RMSE_train

def draw_experiment(setup, theta,device):
    fig = setup.makePlot(theta,device)
    tempdir = tempfile.TemporaryDirectory()
    fig.savefig(tempdir.name + '/plot_train.png', dpi=2*fig.dpi)
    mlflow.log_artifact(tempdir.name + '/plot_train.png')
    plt.close(fig)
    fig = setup.makePlotCI(theta,device)
    fig.savefig(tempdir.name + '/plot_train_CI.png', dpi=2*fig.dpi)
    mlflow.log_artifact(tempdir.name + '/plot_train_CI.png')
    plt.close(fig)
    
    
def save_model(model):
    tempdir = tempfile.TemporaryDirectory()
    torch.save({'state_dict': model.state_dict()}, tempdir.name + '/model.pt')
    mlflow.log_artifact(tempdir.name + '/model.pt')

def save_params_ens(theta):
    tempdir = tempfile.TemporaryDirectory()
    torch.save(theta, tempdir.name + '/theta.pt')
    mlflow.log_artifact(tempdir.name + '/theta.pt')


class AbstractRegressionSetup():
    def __init__(self,):
        self.experiment_name=''
        self.plot = False
        self.param_count=None
        self.device=None
        self.sigma_noise=None
        self.n_train_samples=None
        self.sigma_prior=None
        self.seed=None
        self.input_dim=None
        self.test_ratio=test_ratio

    #@abstractmethod

    # def logposterior(self):
    #     raise NotImplementedError('subclasses must override logposterior()')

    def makePlot(self):
        if self.plot:
            raise NotImplementedError('subclasses with plot=True must override makePlot()')

    def evaluate_metrics(self, theta, device='cpu'):
        theta = theta.to(device)
        
        y_pred=self._model(self._X_test.to(device), theta)
        LPP_test = LPP(y_pred, self._y_test.to(device), self.sigma_noise, device)

        std_y_train = torch.tensor(1.)
        if hasattr(self, '_scaler_y'):
            std_y_train=torch.tensor(self._scaler_y.scale_, device=device).squeeze().float()
        
        y_pred_mean = y_pred.mean(dim=0)
        RMSE_test = RMSE(y_pred_mean, self._y_test.to(device), std_y_train, device)
        
        y_pred_train_mean=self._model(self._X_train.to(device), theta).mean(dim=0)
        RMSE_train = RMSE(self._y_train.to(device), y_pred_train_mean, std_y_train, device)
        
        PICP_test=PICP(y_pred, self._y_test.to(device), device)

        MPIW_test= MPIW(y_pred, device, std_y_train)
        return LPP_test, RMSE_test, RMSE_train, PICP_test, MPIW_test

    def _logprior(self, theta):
        return logmvn01pdf(theta, self.device, v=self.sigma_prior)

    def _normalized_prediction(self, X, theta, device):
        """Predict raw INVERSE normalized values for M models on N data points of D-dimensions
		Arguments:
			X {[tensor]} -- Tensor of size NxD
			theta {[type]} -- Tensor[M,:] of models

		Returns:
			[tensor] -- MxNx1 tensor of predictions
		"""
        assert type(theta) is torch.Tensor
        y_pred = self._model(X.to(device), theta)
        if hasattr(self, '_scaler_y'):
            y_pred = y_pred * torch.tensor(self._scaler_y.scale_, device=device).float() + torch.tensor(self._scaler_y.mean_, device=device).float()
        return y_pred

    def _loglikelihood(self, theta, X, y, device):
        """
		parameters:
			theta (Tensor): M x param_count (models)
			X (Tensor): N x input_dim
			y (Tensor): N x 1
		output:
			LL (Tensor): M x N (models x data)
		"""
        y_pred = self._model(X.to(device), theta) # MxNx1 tensor
        #y_pred = self._normalized_prediction(X, theta, device)  # MxNx1 tensor
        return NormalLogLikelihood(y_pred, y.to(device), self.sigma_noise)

    def logposterior(self, theta):
        return self._logprior(theta) + torch.sum(self._loglikelihood(theta, self._X_train, self._y_train, self.device),dim=1)
    
    def _split_holdout_data(self):
        X_tv, self._X_test, y_tv, self._y_test = train_test_split(self._X, self._y, test_size=test_ratio, random_state=self.seed) #.20
        self._X_train, self._y_train = X_tv, y_tv
        #self._X_train, self._X_validation, self._y_train, self._y_validation = train_test_split(X_tv, y_tv, test_size=0.25, random_state=seed)
        

    def _normalize_data(self):        
        self._scaler_X, self._scaler_y = fitStandardScalerNormalization(self._X_train, self._y_train)
        self._X_train, self._y_train = normalize(self._X_train, self._y_train, self._scaler_X, self._scaler_y)
        #self._X_validation, self._y_validation = normalize(self._X_validation, self._y_validation, self._scaler_X, self._scaler_y)
        self._X_test, self._y_test = normalize(self._X_test, self._y_test, self._scaler_X, self._scaler_y)

    def _flip_data_to_torch(self):
        self._X = torch.tensor(self._X, device=self.device).float()
        self._y = torch.tensor(self._y, device=self.device).float()
        self._X_train = torch.tensor(self._X_train, device=self.device).float()
        self._y_train = torch.tensor(self._y_train, device=self.device).float()
        #self._X_validation = torch.tensor(self._X_validation, device=self.device).float()
        #self._y_validation = torch.tensor(self._y_validation, device=self.device).float()
        self._X_test = torch.tensor(self._X_test, device=self.device).float()
        self._y_test = torch.tensor(self._y_test, device=self.device).float()
        self.n_train_samples=self._X_train.shape[0]



    def loglikelihood(self, theta, batch_size=None):
        if batch_size is None:
            batch_size=self.n_train_samples
        index=torch.randperm(self._X_train.shape[0])
        X_train=self._X_train[index][0:batch_size]
        y_train=self._y_train[index][0:batch_size]
        ll=torch.sum(self._loglikelihood(theta, X_train, y_train, self.device),dim=1)
        return ll

    
    def projection(self,theta0,theta1, n_samples, ratio_ood):
        #compute size of both samples
        #n_samples=self.n_train_samples
        n_ood=int(ratio_ood*n_samples)
        n_id=n_samples-n_ood
        
        #batch sample from train
        index=torch.randperm(self._X_train.shape[0])
        X_id=self._X_train[index][:n_id]
            
        #batch sample OOD    
        epsilon=0.1
        M = self._X_train.max(0, keepdim=True)[0]+epsilon
        m = self._X_train.min(0, keepdim=True)[0]-epsilon
        X_ood = torch.rand(n_ood,self.input_dim).to(self.device) * (M-m) + m    
        
        X=torch.cat([X_id, X_ood])
        
        #compute projection on both paramters with model
        theta0_proj=self._model(X, theta0).squeeze(2)
        theta1_proj=self._model(X, theta1).squeeze(2)
        return theta0_proj, theta1_proj

    def projection_normal(self,theta0,theta1, n_samples, sigma=1.):
        #compute size of both samples
        #n_samples=self.n_train_samples
             
        #batch sample from train
        train_inputs=self._X_train
        if train_inputs.shape[0] < n_samples:
            train_inputs=torch.cat([train_inputs]*int(n_samples/train_inputs.shape[0]+1.))
        index=torch.randperm(train_inputs.shape[0])
            
        X = train_inputs[index][:n_samples]+ sigma*torch.rand(n_samples,self.input_dim).to(self.device)     
        #compute projection on both paramters with model
        theta0_proj=self._model(X, theta0).squeeze(2)
        theta1_proj=self._model(X, theta1).squeeze(2)
        return theta0_proj, theta1_proj

        
    def train_data(self):
        return self._X_train, self._y_train
    
    def test_data(self):
        return self._X_test, self._y_test
        
    
    # @abstractmethod
    # def evaluate(self):
    #     raise NotImplementedError('subclasses must override evaluate()')

    # @abstractmethod
    # def get_logposterior(self):
    #     raise NotImplementedError('subclasses must override get_logposterior()')