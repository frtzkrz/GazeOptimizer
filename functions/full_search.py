import h5py


def full_search(self):
    self.fill_cache_from_h5py()
    return self.optimize()
