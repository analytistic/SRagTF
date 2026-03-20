from torch.utils.data import Dataset
from pathlib import Path




class BaseDataset(Dataset):
    def __init__(
        self,
        datasets: str,
        data_path: str,
        **kwargs
    ):
        super().__init__()
        self.datasets = datasets
        self.data_path = Path(data_path)


        




    