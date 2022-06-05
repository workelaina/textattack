import pickle
def write_pkl(content, path, print_option=False):
    '''write content on path with path
    Dependency : pickle
    Args:
        content - object to be saved
        path - string
                ends with pkl
    '''
    with open(path, 'wb') as f:
        if print_option:
            print("Pickle is written on %s"%path)
        try: pickle.dump(content, f)
        except OverflowError: pickle.dump(content, f, protocol=4)

def read_pkl(path, encoding='ASCII', print_option=False):
    '''read path(pkl) and return files
    Dependency : pickle
    Args:
        path - string
               ends with pkl
    Return:
        pickle content
    '''
    if print_option:
        print("Pickle is read from %s"%path)
    with open(path, 'rb') as f: return pickle.load(f, encoding=encoding)
