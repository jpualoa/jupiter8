import os

def valid_path(pathname):
    pathname = os.path.expandvars(os.path.expanduser(pathname))
    if not os.path.exists(pathname):
        raise ValueError("%s does not exist" % pathname)
    return pathname
