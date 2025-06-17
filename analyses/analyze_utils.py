import os
import pandas as pd

class PathLoader:
    def __init__(self, dir_path: str, k2in: list=[], k2ex: list=[], sort: bool=True):
        self.dir_path = dir_path
        self.k2in = k2in
        self.k2ex = k2ex
        self.sort = sort

    def get_file_names(self, sort: bool=True)-> list:
        file_names = os.listdir(self.dir_path)
        if len(self.k2in) > 0:
            # Include only filenames that contain any of the strings in k2in
            file_names = [file_name for file_name in file_names if any(k in file_name for k in self.k2in)]
        if len(self.k2ex) > 0:
            # Filter out filenames that contain any of the strings in k2ex 
            file_names = [file_name for file_name in file_names if not any(k in file_name for k in self.k2ex)]
        if sort:
            file_names.sort()

        self.file_names = file_names
        return file_names

    def get_file_paths(self, sort=True):
        file_names = self.get_file_names(sort)
        file_paths = [os.path.join(self.dir_path, file_name) for file_name in file_names]
        
        self.file_paths = file_paths
        return file_paths

    def get_dfs(self, sort=True):
        file_paths = self.get_file_paths(sort)
        dfs = [pd.read_csv(file_path) for file_path in file_paths]
        
        self.dfs = dfs
        return dfs
    
# Test
if __name__ == "__main__":
    dir_path = '/home/hwkang/dev-TSB-AD/TSB-AD/results/daphnet'
    k2in = ['293']
    k2ex = ['303']

    loader = PathLoader(dir_path=dir_path, k2in=k2in, k2ex=k2ex, sort=True)

    file_names = loader.get_file_names()
    print("File Names:", file_names)

    file_paths = loader.get_file_paths()
    print("File Paths:", file_paths)

    dfs = loader.get_dfs()
    print("DataFrames:", [df.head() for df in dfs])