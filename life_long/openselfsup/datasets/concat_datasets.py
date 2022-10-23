import torch


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ScaledConcatDataset(torch.utils.data.Dataset):
    def __init__(self, scale_ratio=2, *datasets):
        self.datasets = datasets
        assert isinstance(scale_ratio, int)
        for _dataset in datasets[1:]:
            assert len(_dataset) >= len(datasets[0]) * scale_ratio
        self.scale_ratio = scale_ratio

    def __getitem__(self, i):
        curr_data = [self.datasets[0][i]]
        for _dataset in self.datasets[1:]:
            for _idx in range(self.scale_ratio):
                curr_data.append(
                        _dataset[i*self.scale_ratio + _idx])
        return tuple(curr_data)

    def __len__(self):
        return len(self.datasets[0])


class ConcatLoader:
    def __init__(self, *loaders):
        self.loaders = loaders

    def __len__(self):
        return min(len(d) for d in self.loaders)

    def __iter__(self):
        sentinel = object()
        iterators = [iter(it) for it in self.loaders]
        while iterators:
            result = []
            for it in iterators:
                elem = next(it, sentinel)
                if elem is sentinel:
                    return
                result.append(elem)
            yield result
