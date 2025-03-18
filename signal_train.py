import torch
import torch.nn as nn
import torch.optim as optim
from stream_handler import StreamManager
from signal_accumulative_mlp import SignalAccumulativeMLP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(model, criterion, optimizer, t_loader, v_loader, n_epochs=10, save_path="models/signal_accumulative_model"):
    for epoch in range(n_epochs):
        model.reset_accumulators()
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in t_loader:
            device = next(model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = train_loss / len(t_loader.dataset)
        train_acc = correct / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in v_loader:
                device = next(model.parameters()).device
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        val_loss = val_loss / len(v_loader.dataset)
        val_acc = correct / total

        logger.info(f'Epoch {epoch + 1}/{n_epochs}: '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if save_path:
            model.save_model(f"{save_path}_epoch_{epoch + 1}.pt")

    return model


def main():
    # Init Stream Manager
    stream_manager = StreamManager()

    # Parameters
    vocab_size = stream_manager.get_vocab_size()
    num_streams = 4
    sequence_length_train = 200
    sequence_length_test = 20
    accumulator_decay = 0.9
    learning_rate = 0.001
    num_epochs = 10
    save_path = "models/signal_accumulative_model"

    # Data Loading
    # Training
    x_batches_train, y_batches_train = stream_manager.get_torch_batches(num_streams,
                                                                        sequence_length=sequence_length_train)
    train_loader = stream_manager.load_data(num_streams, x_batches_train, y_batches_train)

    stream_manager.set_seed(234)

    # Validation
    x_batches_test, y_batches_test = stream_manager.get_torch_batches(num_streams,
                                                                      sequence_length=sequence_length_test)
    validation_loader = stream_manager.load_data(num_streams, x_batches_test, y_batches_test)

    # Signal Model
    model = SignalAccumulativeMLP(vocab_size, num_streams, accumulator_decay=accumulator_decay)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model = train(model,
                  criterion,
                  optimizer,
                  train_loader,
                  validation_loader,
                  n_epochs=num_epochs,
                  save_path=save_path)

if __name__ == '__main__':
    main()
