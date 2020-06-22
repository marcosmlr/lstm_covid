# coding: utf-8

# --------------------------
#        Imports
# --------------------------
import numpy as np
import os
import errno

# Notes - numpy.seterr
#
# The floating-point exceptions are defined in the IEEE 754 standard [1]:
#
#     Division by zero: infinite result obtained from finite numbers.
#     Overflow: result too large to be expressed.
#     Underflow: result so close to zero that some precision was lost.
#     Invalid operation: result is not an expressible number, typically indicates that a NaN was produced.
#
# [1]	http://en.wikipedia.org/wiki/IEEE_754
#
np.seterr(under='ignore')


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

