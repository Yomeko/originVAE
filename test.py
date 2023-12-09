import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple

def test(*args, a=0.00025):

    b=args[0]
    c=args[1]

    print(a)
    print(b)
    print(c)


if __name__ == '__main__':

    a=torch.randn(1,3)
    b=torch.randn(2,2,3)
    c=torch.randn(1,3)

    d=[a,b,c]

    test(*d)
    print(torch.cuda.is_available())