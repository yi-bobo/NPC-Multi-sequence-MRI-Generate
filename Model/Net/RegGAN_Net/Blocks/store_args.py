import functools
import inspect
import torch

if not hasattr(inspect, 'getargspec'):
    inspect.getargspec = inspect.getfullargspec
def store_config_args(func):
    """
    Class-method decorator that saves every argument provided to the
    function as a dictionary in 'self.config'. This is used to assist
    model loading - see LoadableModel.
    """

    out = inspect.getargspec(func)
    if len(out)==2:
        attrs, defaults = out[0], out[1]
    elif len(out)==4:
        attrs, varargs, varkw, defaults = out[0], out[1],out[2], out[3]

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.config = {}

        # first save the default values
        if defaults:
            for attr, val in zip(reversed(attrs), reversed(defaults)):
                self.config[attr] = val

        # next handle positional args
        for attr, val in zip(attrs[1:], args):
            self.config[attr] = val

        # lastly handle keyword args
        if kwargs:
            for attr, val in kwargs.items():
                self.config[attr] = val

        return func(self, *args, **kwargs)
    return wrapper

def torch2hwcuint8(x, clip=False):
    if clip:
        x = torch.clamp(x, -1, 1)
    x = (x + 1.0) / 2.0
    return x