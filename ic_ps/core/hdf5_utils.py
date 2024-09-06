import numpy as np
import pandas as pd

def h5store(filename, df, metadata):

    store = pd.HDFStore(filename)
    store.put('data', df)
    store.get_storer('data').attrs.metadata = metadata 
    store.close()

def h5load(store, table='data'):
    data = store[table]
    metadata = store.get_storer(table).attrs.metadata
    return data, metadata
