import os
from multiprocessing import Pool

if __name__ == "__main__":

    os.system('python runs_Yann-GeN.py')
    
    os.system('python -m runs_Yann-Fun-ratioOOD --ratio_ood=1.')

    os.system('python -m runs_Yann-Fun-ratioOOD --ratio_ood=.75')
    
    os.system('python -m runs_Yann-Fun-ratioOOD --ratio_ood=.5')
    
    
    """
    def run_dataset(algorithm):                                                             
        os.system('python {}'.format(algorithm))                                                                                                         
    pool = Pool(processes=1) 
    
    for dataset in ['foong', 'foong_mixed', 'foong_sparse']:
        pool.map(run_dataset, ["-m Experiments.GeNNeVI-mr --nb_models=5 --setup="+dataset+" --n_samples_KL=500 --device='cuda:0'"])
        print(dataset+': done :-)')

    
    python -m runs_Yann-Fun-ratioOOD.py --ratio_ood=1.

    python -m runs_Yann-Fun-ratioOOD.py --ratio_ood=.8

    python -m runs_Yann-Fun-ratioOOD.py --ratio_ood=.5

    python runs_Yann-GeN.py


    """

