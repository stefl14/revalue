import click
import logging
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping  # Import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from dataset import MultiSpectralDataModule
from model import MultiSpectralModel
from config import config

# Set up basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@click.command()
@click.option('--dataset_path', default=config["dataset_path"], help='Path to the dataset')
@click.option('--num_epochs', default=config["num_epochs"], type=int, help='Number of epochs to train')
@click.option('--batch_size', default=config["batch_size"], type=int, help='Batch size for training')
@click.option('--learning_rate', default=config["learning_rate"], type=float, help='Learning rate')
@click.option('--num_input_channels', default=config["num_input_channels"], type=int, help='Number of input channels')
def main(dataset_path, num_epochs, batch_size, learning_rate, num_input_channels):
    for init_num in range(1, 6):  # Loop for 5 different random inits
        logging.info(f"Starting training process for initialization {init_num}.")
        seed_everything(init_num)  # Set a new seed for each initialization

        # Adjust logger and checkpoint for each init
        logger = TensorBoardLogger("tb_logs", name=f"my_model_init_{init_num}")
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"my_model_checkpoints/init_{init_num}",
            filename="model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=3,
            mode="min",
        )

        # Define the EarlyStopping callback
        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            patience=4,
            verbose=True,
            mode="min"
        )

        num_classes = 10  # Hardcoded for now, but should be determined from the dataset

        data_module = MultiSpectralDataModule(dataset_path=dataset_path, batch_size=batch_size)
        data_module.setup(stage='fit')

        model = MultiSpectralModel(num_input_channels=num_input_channels, learning_rate=learning_rate,
                                   num_classes=num_classes, class_names=data_module.class_names)

        trainer = Trainer(
            logger=logger,
            callbacks=[checkpoint_callback, early_stopping_callback],  # Add the early_stopping_callback
            max_epochs=num_epochs,
        )

        logging.info(f"Trainer configured for init {init_num}. Starting training...")
        trainer.fit(model, datamodule=data_module)
        logging.info(f"Training for init {init_num} completed. Starting testing...")
        trainer.test(model, datamodule=data_module)
        logging.info(f"Testing for init {init_num} completed.")


if __name__ == "__main__":
    main()
