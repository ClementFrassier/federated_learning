# Federated Learning with Flower v2 and PyTorch

This repository demonstrates how to set up a modern Federated Learning environment using the **Flower v2** architecture. It uses `ClientApp` and `ServerApp` constructs for granular task orchestration, `flwr-datasets` for dynamic data partitioning, and trains a CNN model on the **Fashion MNIST** dataset.

## Architecture

This project adopts the native Flower v2 structure (Mods API), preparing it for real-world deployments and advanced simulation:
- **`task.py`**: Defines the PyTorch neural network, training/testing functions, and loads/partitions the `fashion_mnist` dataset dynamically using `flwr_datasets.FederatedDataset`.
- **`client_app.py`**: Contains the `ClientApp` logic using `@app.train()` and `@app.evaluate()` decorators. Communicates via `Message` and `RecordDict`.
- **`server_app.py`**: Contains the `ServerApp` logic using the `@app.main()` decorator. It dynamically reads configurations from `context.run_config` and runs the `FedAvg` strategy.
- **`pyproject.toml`**: Defines project dependencies and Flower run configurations (number of rounds, learning rate, batch size, etc.).

## Requirements

Ensure you have Python 3.8+ installed. You can install the necessary dependencies using pip from the `pyproject.toml` definition:

```bash
pip install -e .
```

*Note: The primary dependencies are `flwr[simulation]`, `flwr-datasets[vision]`, `datasets`, `torch`, and `torchvision`.*

## Running the Simulation

You no longer need to launch the server and clients in separate terminal windows. Flower Simulation engine automatically orchestrates the federated learning process based on your `pyproject.toml` configuration.

To start the simulation, navigate to this directory and simply run:

```bash
flwr run .
```

### Configuration

You can easily adjust hyperparameters without touching the Python code. Open `pyproject.toml` and modify the `[tool.flwr.app.config]` section to change:
- `num-server-rounds`: The total number of federated rounds.
- `batch-size`: The batch size used for local client training.
- `local-epochs`: Number of local epochs clients train for each round.
- `learning-rate`: The learning rate used by the optimizer.

After the simulation finishes, the global model weights are saved in the current directory as `final_model.pt`.
