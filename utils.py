from pytorch_lightning.loggers import TensorBoardLogger

# Create a TensorBoardLogger
logger = TensorBoardLogger("tb_logs", name="my_model")

from torchvision.utils import make_grid
from pytorch_lightning.callbacks import Callback

import random

import torch


class ImagePredictionLogger(Callback):
    def __init__(self, num_samples=3):
        super().__init__()
        self.num_samples = num_samples

    def on_test_epoch_end(self, trainer, pl_module):
        test_loader = trainer.test_dataloaders
        if isinstance(test_loader, list):
            test_loader = test_loader[0]  # If it's a list, take the first DataLoader
        test_dataset = test_loader.dataset

        sample_indices = random.sample(range(len(test_dataset)), self.num_samples)
        samples = [test_dataset[i] for i in sample_indices]

        for i, (image, label_index) in enumerate(samples):
            # Assuming image is a numpy array with shape [C, H, W] and C >= 3

            # Select three channels for visualization
            if image.shape[0] >= 3:
                # Example: Selecting three specific channels for RGB visualization
                image_tensor = torch.tensor(
                    image[[2, 1, 0], :, :], dtype=torch.float32
                )  # Adjust channel indices as needed
            else:
                # Fallback for single-channel images or other cases
                image_tensor = torch.tensor(image, dtype=torch.float32)

            # Ensure image tensor is in [B, C, H, W] format for make_grid
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)

            # Convert to a grid of images (useful if logging multiple images)
            img_grid = make_grid(image_tensor)

            # Log the image grid to TensorBoard
            trainer.logger.experiment.add_image(
                f"sample_{i}_label_{label_index}",
                img_grid,
                global_step=trainer.global_step,
            )
