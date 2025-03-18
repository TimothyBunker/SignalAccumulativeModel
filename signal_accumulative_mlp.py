import os
import torch
import torch.nn as nn


class SignalAccumulativeMLP(nn.Module):
    def __init__(self, vocab_size, num_streams, hidden_size1=128, hidden_size2=96, hidden_size3=64, num_classes=3, accumulator_decay=0.9):
        super(SignalAccumulativeMLP, self).__init__()

        self.num_streams = num_streams # alternative name for batch size
        self.input_size = vocab_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.hidden_size3 = hidden_size3
        self.output_size = num_classes

        self.accumulator_decay = accumulator_decay
        self.register_buffer("signal_accumulators", torch.zeros(num_streams, vocab_size))

        # TODO add more regularization because we are approaching a deep neural network range
        self.layers = nn.Sequential(
            nn.Linear(vocab_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size3, num_classes)
        )

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

    def save_model(self, path='models/signal_model_v1.pt'):
        """
        Saves the model architecture, weights, and accumulator states

        Args:
            path (str): Path where model should be saved
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
            }
        }

        torch.save(save_dict, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path='models/signal_model_v1.pt', device=None):
        """
        Loads and instantiates a model from the given path

        Args:
            path (str): Path to the saved model file
            device (str, optional): Device to load model to ('cpu', 'cuda')

        Returns:
            SignalAccumulativeMLP: Loaded model instance
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

            # Set model to evaluation mode
            model.eval()

            # Move to specified device
            model = model.to(device)

            print(f"Model loaded from {path} to {device}")
            return model

        except FileNotFoundError:
            print(f"Error: Model file {path} not found")
            raise
        except Exception as e:
            print(f"Error loading model: {e}")
            raise