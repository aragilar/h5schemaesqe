# -*- coding: utf-8 -*-
"""
Useful functions
"""

def allvars(obj):
    try:
        return vars(obj)
    except TypeError as e:
        try:
            return obj._asdict()
        except AttributeError as f:
            raise e from f # maybe?
