from data_handler import DataAggregator
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class StreamManager:
    def __init__(self, word_list=None, seed=42):
        if word_list is None:
            import nltk
            nltk.download('words', quiet=True)
            from nltk.corpus import words
            word_list = words.words()

        self.data_aggregator = DataAggregator(word_list=word_list)
        self.data = self.data_aggregator.get_data()

        self.vocab = "abcdefghijklmnopqrstuvwxyz"
        self._vocab_size = len(self.vocab)

        (self.word_inputs,
         self.word_targets,
         self.shuffled_word_inputs,
         self.shuffled_word_targets,
         self.blank_input,
         self.blank_target) = self.separate_data()

        self.seed = seed

        self.input_streams = []
        self.target_streams = []

    def set_seed(self, num):
        self.seed = num

    def get_vocab_size(self):
        return self._vocab_size

    def separate_data(self):
        word_inputs = []
        word_targets = []
        shuffled_word_inputs = []
        shuffled_word_targets = []
        blank_input = [[0.0] * self._vocab_size]
        blank_target = [0]

        for word in self.data[0]:
            word_input = []
            word_target = []
            for one_hot, target in word:
                word_input.append(one_hot)
                word_target.append(target)
            word_inputs.append(word_input)
            word_targets.append(word_target)

        # Same for shuffled words
        for word in self.data[1]:
            shuffled_input = []
            shuffled_target = []
            for one_hot, target in word:
                shuffled_input.append(one_hot)
                shuffled_target.append(target)
            shuffled_word_inputs.append(shuffled_input)
            shuffled_word_targets.append(shuffled_target)

        # Convert to NumPy arrays
        word_inputs = np.array(word_inputs, dtype=object)
        word_targets = np.array(word_targets, dtype=object)
        shuffled_word_inputs = np.array(shuffled_word_inputs, dtype=object)
        shuffled_word_targets = np.array(shuffled_word_targets, dtype=object)
        blank_input = np.array(blank_input, dtype=object)
        blank_target = np.array(blank_target, dtype=object)

        return word_inputs, word_targets, shuffled_word_inputs, shuffled_word_targets, blank_input, blank_target

    def generate_stream(self, sequence_length=64, random_state=None):
        """
        :param random_state: a random seed tracked across streams
        :param sequence_length: number of blanks and words
        :return: s_inputs, s_targets: stream numpy array of inputs and targets
        """
        s_inputs = []
        s_targets = []

        if random_state is None:
            rng = np.random.RandomState(self.seed)
        elif isinstance(random_state, int):
            rng = np.random.RandomState(random_state)
        else:
            rng = random_state

        while len(s_inputs) < sequence_length:
            list_choice = rng.choice(['a', 'b', 'c'])
            remaining = sequence_length - len(s_inputs)

            if list_choice == 'a' and remaining >= 2:
                sample_idx = rng.choice(len(self.word_inputs))

                s_inputs.append(self.word_inputs[sample_idx])
                s_targets.append(self.word_targets[sample_idx])

                s_inputs.append(self.blank_input)
                s_targets.append(self.blank_target)

            elif list_choice == 'b' and remaining >= 2:
                sample_idx = rng.choice(len(self.shuffled_word_inputs))
                s_inputs.append(self.shuffled_word_inputs[sample_idx])
                s_targets.append(self.shuffled_word_targets[sample_idx])

                s_inputs.append(self.blank_input)
                s_targets.append(self.blank_target)

            elif list_choice == 'c' or remaining == 1:
                s_inputs.append(self.blank_input)
                s_targets.append(self.blank_target)

        return s_inputs, s_targets

    def generate_batches(self, num_streams, sequence_length, as_numpy=True):
        s_inputs = []
        s_targets = []
        smallest_stream_length = float('inf')

        for i in range(num_streams):
            random_state = self.seed + i
            s_input, s_target = self.generate_stream(sequence_length, random_state=random_state)
            s_input_flattened = self.flatten_inputs(s_input)
            s_target_flattened = self.flatten_targets(s_target)

            if len(s_input_flattened) < smallest_stream_length:
                smallest_stream_length = len(s_input_flattened)

            s_inputs.append(s_input_flattened)
            s_targets.append(s_target_flattened)

        batches_X = [[] for _ in range(smallest_stream_length)]
        batches_Y = [[] for _ in range(smallest_stream_length)]

        for j in range(smallest_stream_length):
            for stream_idx in range(num_streams):
                batches_X[j].append(s_inputs[stream_idx][j])
                batches_Y[j].append(s_targets[stream_idx][j])

        if as_numpy:
            batches_X = [np.array(batch) for batch in batches_X]
            batches_Y = [np.array(batch) for batch in batches_Y]

        print(f'Stream Inputs [generate_batches]:\n {s_inputs}')
        print(f'Stream Targets [generate_batches]:\n {s_targets}')

        return batches_X, batches_Y

    def get_torch_batches(self, num_streams, sequence_length):
        batches_X, batches_Y = self.generate_batches(num_streams, sequence_length, as_numpy=True)

        # Convert to PyTorch tensors
        torch_batches_X = [torch.FloatTensor(batch) for batch in batches_X]
        torch_batches_Y = [torch.LongTensor(batch) for batch in batches_Y]

        return torch_batches_X, torch_batches_Y

    @staticmethod
    def load_data(num_streams, tensor_batches_x, tensor_batches_y):
        dataset = TensorDataset(torch.cat(tensor_batches_x), torch.cat(tensor_batches_y))
        dataloader = DataLoader(dataset, batch_size=num_streams, shuffle=False)

        # Print shapes to verify
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            print(f'Batch {batch_idx + 1}')
            print(f'Inputs shape: {inputs.shape}')
            print(f'Targets shape: {targets.shape}')
            break

        return dataloader

    @staticmethod
    def flatten_inputs(X):
        flat_inputs = []
        for word in X:
            for char in word:
                flat_inputs.append(char)
        flat_inputs = np.array(flat_inputs, dtype=np.float32)

        return flat_inputs

    @staticmethod
    def flatten_targets(Y):
        flat_targets = []
        for target_group in Y:
            for target in target_group:
                flat_targets.append(target)

        flat_targets = np.array(flat_targets, dtype=np.int16)

        return flat_targets
