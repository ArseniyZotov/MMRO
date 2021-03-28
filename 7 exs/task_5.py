import functools

def check_arguments(*types):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args):
            if len(types) > len(args):
                raise TypeError
            for required_type, value in zip(types, args):
                if not isinstance(value, required_type):
                    raise TypeError
            return f(*args)
        return wrapper
    return decorator
