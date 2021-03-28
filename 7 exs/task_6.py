import functools
import inspect

def substitutive(f):
    
    @functools.wraps(f)
    def wrapper(*args):
        arg_list = wrapper.arg_list + list(args)
        if (len(arg_list) < wrapper.arg_num):
            ans = substitutive(f)
            ans.arg_list = arg_list
            return ans
        else:
            return f(*(arg_list))
                
    wrapper.arg_list = []
    wrapper.arg_num = len(inspect.getargspec(f).args)
    return wrapper
