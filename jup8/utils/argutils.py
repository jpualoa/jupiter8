import os

def valid_path(pathname):
    pathname = os.path.expandvars(os.path.expanduser(pathname))
    if not os.path.exists(pathname):
        raise ValueError("%s does not exist" % pathname)
    return pathname


def valid_list(val, n=None):
    """Returns 'val', a comma-delimited string, as a list of strings or raises a
    ValueError"""
    try:
        if type(val) is not list: val = [str(x) for x in val.split(',')]
    except (ValueError, TypeError): raise ValueError("Invalid value: %s" %val)

    if n and len(val) != n:
        raise ValueError("Argument requires %s elements" %n)
    return val
