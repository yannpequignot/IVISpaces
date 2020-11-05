import os
from multiprocessing import Pool

if __name__ == "__main__":

    def run_script(algorithm):                                                             
        os.system('python -m {}'.format(algorithm))                                                                                                         
    pool = Pool(processes=1) 

    print('Exp2 large')
    pool.map(run_script, ["UCI_Exp2 --set=large"])  
#     print('Exp1')
#     pool.map(run_script, ["UCI_Exp1"])  
#     print('Exp2 small')
#     pool.map(run_script, ["UCI_Exp2 --set=small"])  

    
    

