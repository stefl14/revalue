from pytorch_lightning.loggers import TensorBoardLogger


# Create a TensorBoardLogger
logger = TensorBoardLogger("tb_logs", name="my_model")

from rasterio import open as rio_open

from typing import Optional, Literal

from torch import Tensor
from torch.utils.data import DataLoader

from torchgeo.transforms import AugmentationSequential

import pytorch_lightning as pl

import os
import numpy as np
from torch.utils.data import Dataset
import kornia.augmentation as K
from typing import Tuple, List


class SimpleScaling(K.IntensityAugmentationBase2D):
    """
    Scales input images to a [0, 1] range by dividing by 10000.
    This adjustment is intended to match the preprocessing approach of certain pre-trained models.
    """

    def __init__(self):
        # Initialize with a probability of 1 to always apply this transformation.
        super().__init__(p=1)

    def apply_transform(self, input: Tensor) -> Tensor:
        """
        Apply scaling transformation to the input tensor.

        Args:
            input (Tensor): The input image tensor.

        Returns:
            Tensor: The scaled image tensor.
        """
        return input / 10000.0

class MultiSpectralDataset(Dataset):
    """
    Initializes the dataset.

    Args:
        root_dir (str): Directory with all the images.
        subset (str): One of 'train', 'val', or 'test'.
        split_ratios (tuple): Ratios to split dataset into training, validation, and testing.
        use_data_augmentation (bool): Whether to use data augmentation.
    """
    def __init__(self, root_dir: str, subset: str = "train", split_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1), use_data_augmentation: bool = True):
        """
        Initializes the dataset.

        Args:
            root_dir (str): Directory with all the images.
            subset (str): One of 'train', 'val', or 'test'.
            split_ratios (tuple): Ratios to split dataset into training, validation, and testing.
            use_data_augmentation (bool): Whether to use data augmentation.
        """
        self.root_dir = root_dir
        self.subset = subset
        self.samples = self._load_samples()
        self.label_to_index = self._get_label_to_index()
        self._split_dataset(
            split_ratios
        )
        self.use_data_augmentation = use_data_augmentation

        if self.use_data_augmentation:
            self.augmentations = AugmentationSequential(
                SimpleScaling(),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                K.RandomAffine(degrees=(0, 90), p=0.25),
                K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.25),
                K.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0), p=0.25),
                data_keys=["image"],
            )
        else:
            self.augmentations = None  # Or define a default set of transformations

        if self.subset == "train":
            self.current_samples = self.train_samples
        elif self.subset == "val":
            self.current_samples = self.val_samples
        elif self.subset == "test":
            self.current_samples = self.test_samples
        else:
            raise ValueError("Subset must be 'train', 'val', or 'test'")

    def _load_samples(self) -> List[Tuple[str, str]]:
        """Loads samples from the root directory."""
        samples = []
        for category_dir in filter(lambda x: os.path.isdir(os.path.join(self.root_dir, x)), os.listdir(self.root_dir)):
            label = category_dir
            label_dir = os.path.join(self.root_dir, category_dir)
            for image_file in filter(lambda x: x.endswith((".tif", ".jpg")), os.listdir(label_dir)):
                image_path = os.path.join(label_dir, image_file)
                samples.append((image_path, label))
        return samples

    def _get_label_to_index(self) -> dict:
        """Creates a mapping from label to index."""
        label_set = sorted({label for _, label in self.samples})
        return {label: idx for idx, label in enumerate(label_set)}

    def _split_dataset(self, split_ratios):
        np.random.shuffle(self.samples)
        total_samples = len(self.samples)
        train_size = int(total_samples * split_ratios[0])
        val_size = int(total_samples * split_ratios[1])
        self.train_samples = self.samples[:train_size]
        self.val_samples = self.samples[train_size : train_size + val_size]
        self.test_samples = self.samples[train_size + val_size :]

    def __len__(self):
        return len(self.current_samples)

    def __getitem__(self, idx):
        image_path, label = self.current_samples[
            idx
        ]  # Use self.current_samples instead of self.samples
        with rio_open(image_path) as img:
            image = img.read()
            image = image.astype(np.float32)  # Convert image to float32
        label_index = self.label_to_index[label]  # Convert label to index
        return image, label_index
def get_class_names(dataset: MultiSpectralDataset) -> list:
    """
    Gets the list of class names ordered by their corresponding indices.

    Args:
        dataset: An instance of MultiSpectralDataset

    Returns:
        list: A list of class names ordered by their indices.
    """
    index_to_label = {v: k for k, v in dataset.label_to_index.items()}
    class_names = [index_to_label[i] for i in range(len(index_to_label))]
    return class_names

class MultiSpectralDataModule(pl.LightningDataModule):
    def __init__(self, dataset_path: str, batch_size: int = 32) -> None:
        """Initializes the data module with paths and batch size.

        Args:
            dataset_path: The file path to the dataset directory.
            batch_size: The size of each data batch.
        """
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.class_names = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Sets up datasets for training, validation, and testing stages.

        Args:
            stage: The stage of the model for which to setup the datasets.
        """
        if stage in {"fit", None}:
            self.train_dataset = self._create_dataset(subset="train")
            self.val_dataset = self._create_dataset(subset="val")
            # Assuming get_class_names is a function that extracts class names from the dataset
            self.class_names = get_class_names(self.train_dataset)

        if stage in {"test", None}:
            self.test_dataset = self._create_dataset(subset="test")

    def _create_dataset(self, subset: Literal['train', 'test', 'validation']) -> MultiSpectralDataset:
        return MultiSpectralDataset(root_dir=self.dataset_path, subset=subset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

