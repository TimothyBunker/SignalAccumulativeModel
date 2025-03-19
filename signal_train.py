import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from stream_handler import StreamManager
from signal_accumulative_mlp import SignalAccumulativeMLP
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_data(stream_manager, num_streams, sequence_length, seed=None):
    """Generate new training data"""
    start_time = time.time()
    if seed is not None:
        stream_manager.set_seed(seed)

    x_batches, y_batches = stream_manager.get_torch_batches(
        num_streams, sequence_length=sequence_length
    )
    generation_time = time.time() - start_time

    # Sample verification
    sample_sum = x_batches[0].sum().item()

    return x_batches, y_batches, generation_time, sample_sum


def analyze_class_distribution(y_batches):
    """Analyze and log class distribution"""
    if isinstance(y_batches, list):
        all_labels = []
        for batch in y_batches:
            if isinstance(batch, torch.Tensor):
                all_labels.extend(batch.cpu().numpy().flatten())
            else:
                all_labels.extend(np.asarray(batch).flatten())
        class_counts = np.bincount(np.asarray(all_labels, dtype=int), minlength=4)
    else:
        class_counts = np.bincount(y_batches.cpu().numpy().flatten(), minlength=4)

    class_pct = class_counts / class_counts.sum() * 100
    return class_counts, class_pct


def verify_data_change(train_loader, last_batch_checksum):
    """Verify that new data is different from previous iteration"""
    first_batch = next(iter(train_loader))[0]
    batch_checksum = torch.sum(first_batch).item()

    changed = True
    if last_batch_checksum is not None:
        if abs(batch_checksum - last_batch_checksum) < 0.01:
            logger.warning("⚠️ Dataset may be identical to previous iteration! Checksums too similar.")
            changed = False
        else:
            logger.info(
                f"Dataset change confirmed. New checksum: {batch_checksum:.4f}, Previous: {last_batch_checksum:.4f}")

    return batch_checksum, changed


def train_epoch(model, train_loader, criterion, optimizer, scheduler):
    """Train for one epoch"""
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        device = next(model.parameters()).device
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = correct / total

    return train_loss, train_acc


def validate(model, val_loader, criterion):
    """Run validation and collect metrics"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # For confusion matrix and per-class metrics
    all_targets = []
    all_predictions = []

    model.reset_accumulators()
    with torch.no_grad():
        for inputs, targets in val_loader:
            device = next(model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

            # Collect predictions and targets for metrics
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total

    return val_loss, val_acc, all_targets, all_predictions


def calculate_metrics(all_targets, all_predictions):
    """Calculate and log detailed performance metrics"""
    # Calculate and display confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    logger.info("Confusion Matrix:")
    for row in cm:
        logger.info(" ".join(f"{cell:5d}" for cell in row))

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_targets, all_predictions, zero_division=0
    )

    # Display per-class performance
    logger.info("\nPer-Class Performance:")
    logger.info(f"{'Class':8} {'Precision':10} {'Recall':10} {'F1-Score':10} {'Support':10}")
    for i in range(len(precision)):
        logger.info(f"{i:8d} {precision[i]:10.4f} {recall[i]:10.4f} {f1[i]:10.4f} {support[i]:10d}")

    # Calculate macro averages
    logger.info("\nMacro Averages:")
    logger.info(f"Precision: {np.mean(precision):.4f}")
    logger.info(f"Recall: {np.mean(recall):.4f}")
    logger.info(f"F1-Score: {np.mean(f1):.4f}")

    return precision, recall, f1, support


def train(model, criterion, optimizer, scheduler, t_loader, v_loader, n_epochs=10, reset_accumulators=True, print_metrics=False):
    """Full training loop for multiple epochs"""
    for epoch in range(n_epochs):
        if reset_accumulators:
            model.reset_accumulators()

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, t_loader, criterion, optimizer, scheduler)

        # Run validation
        val_loss, val_acc, all_targets, all_predictions = validate(model, v_loader, criterion)

        # Log basic metrics
        logger.info(f'Epoch {epoch + 1}/{n_epochs}: '
                    f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Calculate and log detailed metrics
        if print_metrics:
            calculate_metrics(all_targets, all_predictions)

    return model

def refresh_validation_data(stream_manager, num_streams, sequence_length_test):
    """Generate truly fresh validation data"""
    logger.info("Generating fresh validation data...")
    x_batches_test, y_batches_test = stream_manager.generate_validation_data(
        num_streams, sequence_length_test
    )
    return stream_manager.load_data(num_streams, x_batches_test, y_batches_test)


def main():
    # Init Stream Manager
    stream_manager = StreamManager()

    # Parameters
    vocab_size = stream_manager.get_vocab_size()
    num_streams = 64
    sequence_length_train = 200
    sequence_length_test = 200
    accumulator_decay = 0.35

    learning_rate = 0.001
    num_epochs_per_dataset = 6
    total_datasets = 100
    reset_accumulators = True
    save_path = "models/sam_temp2"
    load_model = True

    # Validation data
    validation_loader = refresh_validation_data(stream_manager, num_streams, sequence_length_test)

    # Signal Model
    model = SignalAccumulativeMLP(vocab_size, num_streams, accumulator_decay=accumulator_decay, num_classes=4)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # if load_model:
        # model = SignalAccumulativeMLP.load_model(f"{save_path}_dataset_50.pt")

    for data_iteration in range(total_datasets):
        # Generate new data with better seed variation
        x_batches_train, y_batches_train, generation_time, sample_sum = generate_data(
            stream_manager, num_streams, sequence_length_train,
            seed=(43 * data_iteration + 97) % 10000
        )

        logger.info(f"Dataset {data_iteration + 1} generated in {generation_time:.2f}s with checksum: {sample_sum:.4f}")

        # Analyze class distribution
        class_counts, class_pct = analyze_class_distribution(y_batches_train)
        logger.info(
            f"Class distribution: {class_counts} ({class_pct[0]:.1f}%, {class_pct[1]:.1f}%, {class_pct[2]:.1f}%, {class_pct[3]:.1f}%)")

        # Create data loader
        train_loader = stream_manager.load_data(num_streams, x_batches_train, y_batches_train)

        # Refresh validation data occasionally
        if data_iteration % 10 == 0 and data_iteration > 0:
            logger.info("Refreshing validation dataset...")
            x_batches_test, y_batches_test, _, _ = generate_data(
                stream_manager, num_streams, sequence_length_test,
                seed=1000 + data_iteration
            )
            validation_loader = refresh_validation_data(stream_manager, num_streams, sequence_length_test)

        # Create scheduler
        total_steps = len(train_loader) * num_epochs_per_dataset
        scheduler = OneCycleLR(
            optimizer,
            max_lr=0.001,
            total_steps=total_steps,
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=1000.0
        )

        # Train on this dataset
        model = train(
            model, criterion, optimizer, scheduler,
            train_loader, validation_loader,
            n_epochs=num_epochs_per_dataset,
            reset_accumulators=reset_accumulators
        )
        logger.info(f"Completed training on dataset iteration {data_iteration + 1}")

        # Save the model
        if save_path:
            model.save_model(f"{save_path}_dataset_{data_iteration + 1}.pt")


if __name__ == '__main__':
    main()
