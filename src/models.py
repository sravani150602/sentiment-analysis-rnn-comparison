# models.py
# This is where we build the BRAIN ROBOTS! 🧠🤖
# These are the AI models that learn to understand if reviews are happy or sad
# Think of them as students learning to read emotions!

import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    This is a SIMPLE RNN brain 🧠
    RNN = Recurrent Neural Network
    It reads reviews word by word and remembers what it read before
    Like reading a story and remembering the plot!
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=64, 
                 output_dim=1, n_layers=2, dropout=0.3, activation='tanh'):
        """
        This builds the RNN brain with all its parts
        
        vocab_size: How many different words it knows (10,000)
        embedding_dim: How to represent each word as numbers (100 numbers per word)
        hidden_dim: Size of the brain's memory (64)
        output_dim: Output size (1 = positive or negative)
        n_layers: How many layers of thinking (2)
        dropout: How much to forget to avoid memorizing (0.3 = forget 30%)
        activation: How neurons activate (tanh, relu, or sigmoid)
        """
        super(SentimentRNN, self).__init__()
        
        print(f"🏗️ Building Simple RNN with {activation} activation...")
        
        # Embedding layer - converts word numbers into meaningful vectors
        # Like giving each word a personality described by 100 numbers!
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Choose the activation function (how neurons "fire")
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:  # tanh is default
            self.activation = nn.Tanh()
        
        # The RNN layers - the thinking part of the brain!
        # nonlinearity is set to tanh by default in RNN
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Dropout layer - helps prevent memorizing (overfitting)
        self.dropout = nn.Dropout(dropout)
        
        # Final decision layer - decides if review is positive or negative
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        print(f"✅ Simple RNN built successfully!")
    
    def forward(self, text):
        """
        This is how the brain thinks! It processes the review step by step
        
        text: The review (as numbers)
        Returns: A prediction (positive or negative)
        """
        # Convert words to embeddings
        embedded = self.dropout(self.embedding(text))
        
        # Process through RNN
        output, hidden = self.rnn(embedded)
        
        # Take the last output (after reading the whole review)
        hidden = self.dropout(hidden[-1])
        
        # Apply activation
        hidden = self.activation(hidden)
        
        # Make final decision
        return self.fc(hidden)


class SentimentLSTM(nn.Module):
    """
    This is a SMARTER LSTM brain 🧠✨
    LSTM = Long Short-Term Memory
    It's better at remembering important things from earlier in the review
    Like remembering the beginning of a long story!
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=64,
                 output_dim=1, n_layers=2, dropout=0.3, activation='tanh'):
        """
        Builds the LSTM brain (smarter than regular RNN!)
        """
        super(SentimentLSTM, self).__init__()
        
        print(f"🏗️ Building LSTM with {activation} activation...")
        
        # Embedding layer - same as RNN
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Choose activation function
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:  # tanh
            self.activation = nn.Tanh()
        
        # The LSTM layers - SMARTER thinking!
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final decision layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
        print(f"✅ LSTM built successfully!")
    
    def forward(self, text):
        """
        How the LSTM brain thinks through a review
        """
        # Convert words to embeddings
        embedded = self.dropout(self.embedding(text))
        
        # Process through LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Take the last hidden state
        hidden = self.dropout(hidden[-1])
        
        # Apply activation
        hidden = self.activation(hidden)
        
        # Make final decision
        return self.fc(hidden)


class SentimentBiLSTM(nn.Module):
    """
    This is the SMARTEST Bidirectional LSTM brain 🧠🌟
    It reads the review FORWARDS and BACKWARDS!
    Like reading a mystery novel and knowing the ending helps understand the beginning!
    """
    
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_dim=64,
                 output_dim=1, n_layers=2, dropout=0.3, activation='tanh'):
        """
        Builds the Bidirectional LSTM brain (reads both directions!)
        """
        super(SentimentBiLSTM, self).__init__()
        
        print(f"🏗️ Building Bidirectional LSTM with {activation} activation...")
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Choose activation function
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:  # tanh
            self.activation = nn.Tanh()
        
        # Bidirectional LSTM - reads forward AND backward!
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True  # This makes it read both ways! 🔄
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Final decision layer (hidden_dim * 2 because bidirectional)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        
        print(f"✅ Bidirectional LSTM built successfully!")
    
    def forward(self, text):
        """
        How the Bidirectional LSTM thinks (forwards AND backwards!)
        """
        # Convert words to embeddings
        embedded = self.dropout(self.embedding(text))
        
        # Process through bidirectional LSTM
        output, (hidden, cell) = self.lstm(embedded)
        
        # Concatenate the forward and backward hidden states
        # It's like combining what you learned reading left-to-right 
        # AND right-to-left!
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        hidden = self.dropout(hidden)
        
        # Apply activation
        hidden = self.activation(hidden)
        
        # Make final decision
        return self.fc(hidden)


def get_model(model_type='lstm', vocab_size=10000, embedding_dim=100,
              hidden_dim=64, output_dim=1, n_layers=2, dropout=0.3,
              activation='tanh'):
    """
    This is a MODEL FACTORY! 🏭
    It creates whichever brain robot you want!
    
    model_type: Which brain to build ('rnn', 'lstm', or 'bilstm')
    Other parameters: Settings for the brain
    
    Returns: A brand new brain robot ready to learn!
    """
    print(f"\n🎯 Creating a {model_type.upper()} model...")
    
    if model_type.lower() == 'rnn':
        model = SentimentRNN(vocab_size, embedding_dim, hidden_dim,
                           output_dim, n_layers, dropout, activation)
    elif model_type.lower() == 'lstm':
        model = SentimentLSTM(vocab_size, embedding_dim, hidden_dim,
                            output_dim, n_layers, dropout, activation)
    elif model_type.lower() == 'bilstm':
        model = SentimentBiLSTM(vocab_size, embedding_dim, hidden_dim,
                              output_dim, n_layers, dropout, activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'rnn', 'lstm', or 'bilstm'")
    
    # Count how many parameters (brain cells) the model has
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🔢 Total parameters: {total_params:,}")
    print(f"🎓 Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    """
    This tests our models to make sure they work!
    Like a practice run before the real race! 🏁
    """
    print("🧪 Testing all model architectures...\n")
    
    # Test each model type
    for model_type in ['rnn', 'lstm', 'bilstm']:
        print("=" * 70)
        model = get_model(model_type=model_type)
        
        # Create fake data to test (batch of 4 reviews, each 50 words long)
        fake_input = torch.randint(0, 10000, (4, 50))
        
        # Test if model can process the input
        output = model(fake_input)
        print(f"✅ Test passed! Output shape: {output.shape}")
        print()
    
    print("🎉 All models work perfectly!")