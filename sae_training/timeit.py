"""
    This is a util to time the execution of a function.
    
    (Has to be a separate file, if you put it in utils.py you get circular imports; need to find a permanent home for it)
"""

from functools import wraps
import time

def timeit(func):
    """
        Decorator to time a function.
        
        Taken from https://dev.to/kcdchennai/python-decorator-to-measure-execution-time-54hk
    """
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result
    return timeit_wrapper
    