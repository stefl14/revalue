import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torchgeo.models import ResNet18_Weights
from torchgeo.models import resnet18
from torchmetrics import Accuracy, ConfusionMatrix


def adapt_resnet_to_multispectral(num_input_channels: int) -> nn.Module:
    """ Adapt a ResNet18 model to accept multispectral input.

    Args:
        num_input_channels: int: The number of input channels in the multispectral image.

    Returns:
        nn.Module: The adapted ResNet18 model.
    """
    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

    model = resnet18(weights=weights)
    original_first_layer = model.conv1
    model.conv1 = nn.Conv2d(
        num_input_channels,
        original_first_layer.out_channels,
        kernel_size=original_first_layer.kernel_size,
        stride=original_first_layer.stride,
        padding=original_first_layer.padding,
        bias=False,
    )
    return model


class MultiSpectralModel(pl.LightningModule):
    """
    A PyTorch Lightning module for training and evaluating a multispectral image classification model.

    Attributes:
        model (nn.Module): The neural network model.
        criterion (nn.CrossEntropyLoss): The loss function.
        learning_rate (float): The learning rate for the optimizer.
        accuracy (Accuracy): Metric to calculate accuracy.
        confmat (ConfusionMatrix): Metric to calculate the confusion matrix.
        use_data_augmentation (bool): Flag to use data augmentation.
        class_names (list[str]): Names of the classes.
    """

    def __init__(self, num_input_channels: int, class_names: list[str], learning_rate: float = 0.001,
                 num_classes: int = 10, use_data_augmentation: bool = True):
        super().__init__()
        self.model = adapt_resnet_to_multispectral(num_input_channels=num_input_channels)
        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.confmat = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.use_data_augmentation = use_data_augmentation
        self.class_names = class_names

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the model."""
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int):
        """Processes one batch of training data."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int):
        """Processes one batch of validation data."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.confmat.update(preds, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.accuracy, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        """Called at the end of the validation epoch to log the confusion matrix."""
        confmat_tensor = self.confmat.compute()
        if not self.class_names:
            raise ValueError("Class names have not been set in the model.")

        fig = self.plot_conf_matrix(confmat_tensor, self.class_names)
        self.logger.experiment.add_figure("Confusion Matrix/Validation", fig, self.current_epoch)
        self.confmat.reset()

    def test_step(self, batch: tuple, batch_idx: int):
        """Processes one batch of test data."""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        self.confmat.update(preds, y)
        self.log("test_loss", loss)
        self.log("test_acc", self.accuracy, on_step=False, on_epoch=True)

    def on_test_epoch_end(self):
        """Called at the end of the test epoch to log the confusion matrix."""
        confmat_tensor = self.confmat.compute()
        fig = self.plot_conf_matrix(confmat_tensor, self.class_names)
        self.logger.experiment.add_figure("Confusion Matrix/Test", fig, self.current_epoch)
        self.confmat.reset()

    def configure_optimizers(self):
        """Configures the model's optimizers."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def plot_conf_matrix(self, confmat_tensor: torch.Tensor, class_names: list[str]) -> plt.Figure:
        """Generates a matplotlib figure of the confusion matrix."""
        fig, ax = plt.subplots()
        im = ax.imshow(confmat_tensor.cpu().numpy(), interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(title='Confusion Matrix', ylabel='True label', xlabel='Predicted label')

        tick_marks = torch.arange(len(class_names))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

        thresh = confmat_tensor.max() / 2.
        for i in range(confmat_tensor.shape[0]):
            for j in range(confmat_tensor.shape[1]):
                ax.text(j, i, int(confmat_tensor[i, j]), ha="center", va="center",
                        color="white" if confmat_tensor[i, j] > thresh else "black")
        fig.tight_layout()
        return fig