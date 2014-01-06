
def train_test_split(data, n_train):
    """
    Split the data into a 2-tuple containing (data[:n_train], data[n_train:])
    
    Args:
        data - a numpy or scipy object with a shape and supports indexing
        n_train - an int or a float on the interval (0, 1)
                  if int, then split on the first n_train items
                  if float, then split on the first floor(len(data) * n_train) items
    
    Returns:
        a 2-tuple (data[:n_train], data[n_train:])
    
    Raises:
        ValueError if n_train is float and not on the interval (0,1)
        ValueError if n_train is not float or int
    """
    if isinstance(n_train, float):
        if not (n_train > 0 and n_train < 1):
            raise ValueError('n_train is not on the interval (0,1)') 
        m = data.shape[0]
        n_train = int(m * n_train)
    
    if not isinstance(n_train, int):
        raise ValueError('n_train is not of type int or float')
        
    return (data[:n_train], data[n_train:])
