import mmap
import torch
import numpy

class BinaryDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, seq_len):
        self.file_path = file_path
        self.seq_len = seq_len
        self.file = open(self.file_path, 'rb')
        self.file_size = self.file.seek(0, 2)
        self.file.seek(0)
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.num_samples = self.file_size // 8 // self.seq_len
        self.num_extra_ints = self.file_size // 8 % self.seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.num_samples:
            raise IndexError('Index out of range')
        start = idx * self.seq_len * 8
        end = start + self.seq_len * 8
        if end > self.file_size:
            raise IndexError('Index out of range')
        raw_data = self.mmap[start:end]
        data = numpy.frombuffer(raw_data, dtype='int64')
        return torch.from_numpy(data)

    def close(self):
        if self.mmap is not None:
            self.mmap.close()
            self.mmap = None
        if self.file is not None:
            self.file.close()
            self.file = None

