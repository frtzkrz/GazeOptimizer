import h5py


def full_search(self):
    self.fill_cache_from_h5py()
    angles, cost = self.optimize()
    print(f'Optimal Gaze Angle: {angles}')
    print(f'Corresponding Cost: {cost}')
    return angles, cost