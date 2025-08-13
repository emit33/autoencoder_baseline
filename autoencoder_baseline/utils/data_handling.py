from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class TensorDataset_pair_output(Dataset):
    def __init__(self, data_tensor: torch.Tensor):
        self.data_tensor = data_tensor

        # Ensure data tensor is normalised to live in [0,1]. Warning: This ensures that the new maximum is indeed 1
        if self.data_tensor.max() > 1 or self.data_tensor.min() < 0:
            self.data_tensor = (self.data_tensor - self.data_tensor.min()) / (
                self.data_tensor.max() - self.data_tensor.min()
            )

    def __len__(self):
        return len(self.data_tensor)

    def __getitem__(self, idx):

        return self.data_tensor[idx], idx


def get_tensor_data_dataloader(data_dir: Path, batch_size: int = 32):
    data_tensor = torch.load(data_dir / "imgs.pt").float()
    dataset = TensorDataset_pair_output(data_tensor)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return dataloader


def get_train_dataloader(
    data_dir: Path,
    batch_size: int = 32,
    tensor_data: bool = True,
):
    if tensor_data:
        return get_tensor_data_dataloader(data_dir, batch_size)
    else:
        raise ValueError("Non-tensor data deprecated")
        # return get_img_dir_dataloader(data_dir, batch_size, resolution, grayscale)
