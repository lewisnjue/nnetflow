from.engine import Tensor


def is_tensor(obj):
    return isinstance(obj,Tensor)


def numel(obj:Tensor):
    return obj.data.size



