
# README for Your Machine Learning Project

---

## Getting Started

This repository contains a machine learning project designed for multispectral image analysis using PyTorch Lightning. The project is structured to provide an easy-to-use command-line interface for training the model, along with a set of utilities for data processing and model configuration.

### Prerequisites

- Python 3.8 or higher
- [Poetry](https://python-poetry.org/) for dependency management and packaging.

### Installation

1. **Clone the repository** to your local machine.

2. **Install dependencies** using Poetry. Navigate to the project directory and run:

   ```shell
   poetry install
   ```

   This command reads the `pyproject.toml` and `poetry.lock` files to install the necessary dependencies in a virtual environment.

### Configuration

Before running the training, you may need to adjust the configurations according to your dataset and training requirements. Configuration options are available in `config.py`. Review and modify them as necessary to fit your project's needs.

Note, you will also need to create a data directory with eurosat data i.e. data/EuroSAT_MS_Samples

### Training the Model

To train the model, use the CLI provided in `trainer.py`. The CLI supports various options for training customization, such as setting the number of epochs, batch size, and more.

#### Basic Usage

```shell
poetry run python trainer.py [OPTIONS]
```

Replace `[OPTIONS]` with your desired command-line options to customize the training session. The available options include:

- `--dataset_path`: Path to the dataset. Default is specified in `config.py`.
- `--num_epochs`: Number of epochs for training. Default is specified in `config.py`.
- `--batch_size`: Batch size for training. Default is specified in `config.py`.
- `--learning_rate`: Learning rate for the optimizer. Default is specified in `config.py`.
- `--num_input_channels`: Number of input channels for the model. Default is specified in `config.py`.

#### Example Command

To train your model with custom configurations, you can run the following command:

```shell
poetry run python trainer.py --dataset_path "/path/to/dataset" --num_epochs 100 --batch_size 32 --learning_rate 0.001 --num_input_channels 3
```

#### Viewing TensorBoard Logs

The training session logs are saved in the `tb_logs` directory. You can visualize the logs using TensorBoard by running the following command:

```shell
 tensorboard --logdir=tb_logs --port=8080
```

### Additional Modules

- **`dataset.py`**: Defines the data module for handling the dataset.
- **`model.py`**: Contains the definition of the machine learning model.
- **`utils.py`**: Provides additional utilities for data processing and model training.

