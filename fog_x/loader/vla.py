from fog_x.loader.base import BaseLoader
import fog_x
import glob

class VLALoader(BaseLoader):
    def __init__(self, path, split = None):
        super(VLALoader, self).__init__(path)
        self.index = 0
        self.files = glob.glob(self.path, recursive=True)
    
    def _read_vla(self, data_path):
        traj = fog_x.Trajectory(data_path)
        return traj
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index < len(self.files):
            file_path = self.files[self.index]
            self.index += 1
            return self._read_vla(file_path)
        raise StopIteration
    
    def __len__(self):
        return len(self.files)

    