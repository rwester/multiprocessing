# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 11:21:01 2016

@author: westerr
"""
import multiprocessing as mp
import pandas as pd
import numpy as np

def multiprocess(function, iterator, func_args=None, combine=True, n_pool=-1, include_process_num=True):
    """
    Processes data on n number of processors, use n_pool = -1 for all processors

    function is the function to apply to each object in iterator, must except the object to process and process number if include_process_num=True

    func_args is a dictionary of args to feed function
    
    iterator must be list of files or chunks of dataframes
    
    if combine is True will return the combined data
    
    n_pool can be greater than 1 and less than the cpu count or if negative is 1+cpu_count-|n_pool|
    """
    # Initialize pool
    if n_pool < 0:
        pool = mp.Pool(processes=mp.cpu_count()+1+n_pool)
    elif n_pool > 1 and n_pool  <= mp.cpu_count():
        pool = mp.Pool(processes=n_pool)
    else:
        raise ValueError("n_pool is out of range for cpu count!")
    
    # Apply Async (returns process once complete
    if include_process_num == True:
        results = [pool.apply_async(function, args=(obj, process), kwds=func_args) for process, obj in enumerate(iterator)] 
    else:
        results = [pool.apply_async(function, args=(obj), kwds=func_args) for obj in iterator] 

    # Get results and concat into a single dataframe
    if combine == True:
        result = pd.concat([p.get() for p in results])
    else:
        result = [p.get() for p in results]
    return result
    
def multiprocess_df(function, df, chunksize, **kargs):
    """
    Given a dataframe and chunksize will chunk up a dataframe and pass to multiprocess
    
    function is the function to apply to dataframe, must except a dataframe chunk to process and process number if include_process_num=True

    **kargs are optional args for multiprocess
    """
    # Chunk up dataframe
    chunks = [df[x*chunksize:(x+1)*chunksize] for x in xrange(int(np.ceil(df.shape[0] / float(chunksize))))]
    
    # If length matches, pass to multiprocess
    if df.shape[0] == sum([c.shape[0] for c in chunks]):
        return multiprocess(function, chunks, **kargs)
    else:
        raise ValueError("Error separating chunks, length doesn't match!")
        
