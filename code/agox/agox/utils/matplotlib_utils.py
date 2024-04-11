global mpl_backend
mpl_backend = 'Agg'

def get_mpl_backend():
    global mpl_backend
    return mpl_backend

def set_mpl_backend(new_backend):
    global mpl_backend
    mpl_backend = new_backend

def use_agox_mpl_backend():
    import matplotlib
    matplotlib.use(get_mpl_backend())