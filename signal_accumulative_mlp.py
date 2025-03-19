import os
import torch
import torch.nn as nn


class SignalAccumulativeMLP(nn.Module):
    def __init__(self, vocab_size, num_streams, hidden_size1=128, hidden_size2=96, hidden_size3=64,
                 num_classes=4, accumulator_decay=0.9, dropout_rate=0.2, use_layernorm=True):
        super(SignalAccumulativeMLP, self).__init__()

        self.num_streams = num_streams
        self.input_size = vocab_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = num_classes
        self.accumulator_decay = accumulator_decay

        self.register_buffer("signal_accumulators", torch.zeros(num_streams, vocab_size))

        # Build layers dynamically
        layers = [nn.Linear(vocab_size, hidden_size1), nn.ReLU()]

        # First layer
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size1))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Second layer
        layers.append(nn.Linear(hidden_size1, hidden_size2))
        layers.append(nn.ReLU())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size2))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Third layer
        layers.append(nn.Linear(hidden_size2, hidden_size3))
        layers.append(nn.ReLU())
        if use_layernorm:
            layers.append(nn.LayerNorm(hidden_size3))
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_size3, num_classes))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.signal_accumulators.mul_(self.accumulator_decay)
            self.signal_accumulators.add_(x)
            self.signal_accumulators.clamp_(min=0.0, max=1.0)

        return self.signal_accumulators.clone()

    def forward(self, x):
        assert x.shape[0] == self.num_streams, f"Expected batch size {self.num_streams}, got {x.shape[0]}"
        processed_x = self.preprocess(x)

        output = self.layers(processed_x)
        return output

    def reset_accumulators(self):
        self.signal_accumulators.zero_()

    def save_model(self, path='models/signal_model_v1.pt', optimizer=None, scheduler=None, epoch=None):
        """
        Saves the model architecture, weights, and accumulator states with optional training state

        Args:
            path (str): Path where model should be saved
            optimizer (torch.optim.Optimizer, optional): Optimizer to save state
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to save state
            epoch (int, optional): Current epoch number
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

        # Save model configuration and weights
        save_dict = {
            'model_state_dict': self.state_dict(),
            'config': {
                'vocab_size': self.input_size,
                'num_streams': self.num_streams,
                'hidden_size1': self.hidden_size1,
                'hidden_size2': self.hidden_size2,
                'hidden_size3': self.hidden_size3,
                'num_classes': self.output_size,
                'accumulator_decay': self.accumulator_decay
            },
            'is_stateful': True,  # Flag to indicate this is a stateful model
            'training_mode': self.training  # Save whether in training or eval mode
        }

        # Add training state if provided
        if optimizer is not None:
            save_dict['optimizer_state_dict'] = optimizer.state_dict()

        if scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()

        if epoch is not None:
            save_dict['epoch'] = epoch

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path='models/signal_model_v1.pt', device=None, optimizer=None, scheduler=None,
                   reset_accumulators=False):
        """
        Loads and instantiates a model from the given path with optional training state

        Args:
            path (str): Path to the saved model file
            device (str, optional): Device to load model to ('cpu', 'cuda')
            optimizer (torch.optim.Optimizer, optional): Optimizer to load state into
            scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Scheduler to load state into
            reset_accumulators (bool): Whether to reset accumulators after loading

        Returns:
            tuple: (model, optimizer, scheduler, epoch) if training state was saved,
                   or just model otherwise
        """
        try:
            if device is None:
                device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Load saved state
            checkpoint = torch.load(path, map_location=device)

            # Extract configuration
            config = checkpoint['config']

            # Create new model instance
            model = cls(
                vocab_size=config['vocab_size'],
                num_streams=config['num_streams'],
                hidden_size1=config['hidden_size1'],
                hidden_size2=config['hidden_size2'],
                hidden_size3=config['hidden_size3'],
                num_classes=config['num_classes'],
                accumulator_decay=config['accumulator_decay']
            )

            # Load state dict (includes accumulators)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Optionally reset accumulators
            if reset_accumulators:
                model.reset_accumulators()
                print("Signal accumulators have been reset")

            # Set correct training/eval mode
            if checkpoint.get('training_mode', False):
                model.train()
            else:
                model.eval()

            # Move to specified device
            model = model.to(device)

            result = [model]

            # Load optimizer state if requested and available
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                # Move optimizer state to device
                for state in optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)
                result.append(optimizer)

            # Load scheduler state if requested and available
            if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                result.append(scheduler)

            # Add epoch if available
            if 'epoch' in checkpoint:
                result.append(checkpoint['epoch'])

            print(f"Model loaded from {path} to {device}")

            return result[0] if len(result) == 1 else tuple(result)

        except FileNotFoundError:
            print(f"Error: Model file {path} not found")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
