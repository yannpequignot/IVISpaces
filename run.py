import os
from multiprocessing import Pool

if __name__ == "__main__":

    def run_script(algorithm):                                                             
        os.system('python -m {}'.format(algorithm))                                                                                                         
    pool = Pool(processes=1) 

    print('2mean large')
    pool.map(run_script, ["UCI_2Mean --set=large2"])  
    print('2mean small')
    pool.map(run_script, ["UCI_2Mean --set=small2"])  


    
    

