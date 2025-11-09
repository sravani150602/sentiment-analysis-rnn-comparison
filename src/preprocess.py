# preprocess.py
# This is our CLEANING ROBOT! 🧹
# It takes messy movie reviews and makes them neat and tidy
# Like organizing your toys before playing!

import numpy as np
import pickle
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.datasets import imdb

def load_imdb_data():
    """
    This loads the movie reviews from our data folder
    Like opening a book to read stories! 📖
    
    Returns: Training and testing data with labels
    """
    print("📂 Loading movie reviews from data folder...")
    
    # Get the path relative to the project root
    # This works whether we run from main folder or src folder!
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data')
    
    # Load the saved data
    train_data = np.load(os.path.join(data_path, 'train_data.npy'), allow_pickle=True)
    train_labels = np.load(os.path.join(data_path, 'train_labels.npy'), allow_pickle=True)
    test_data = np.load(os.path.join(data_path, 'test_data.npy'), allow_pickle=True)
    test_labels = np.load(os.path.join(data_path, 'test_labels.npy'), allow_pickle=True)
    
    print(f"✅ Loaded {len(train_data)} training reviews")
    print(f"✅ Loaded {len(test_data)} testing reviews")
    
    return train_data, train_labels, test_data, test_labels


def preprocess_data(train_data, test_data, sequence_length=50, max_words=10000):
    """
    This cleans and prepares the movie reviews!
    It makes all reviews the same length (like cutting paper to the same size ✂️)
    
    train_data: Training reviews (raw)
    test_data: Testing reviews (raw)
    sequence_length: How many words to keep in each review (25, 50, or 100)
    max_words: Maximum number of different words to remember (10,000)
    
    Returns: Cleaned and ready-to-use reviews!
    """
    print(f"✂️ Preparing reviews with length {sequence_length} words...")
    
    # Pad sequences - make all reviews the same length
    # If review is too short, add zeros (like adding blank spaces)
    # If review is too long, cut it (like trimming paper)
    train_data_padded = pad_sequences(
        train_data,
        maxlen=sequence_length,
        padding='post',  # Add zeros at the end
        truncating='post'  # Cut from the end if too long
    )
    
    test_data_padded = pad_sequences(
        test_data,
        maxlen=sequence_length,
        padding='post',
        truncating='post'
    )
    
    # Calculate some statistics to report
    train_lengths = [len(x) for x in train_data]
    test_lengths = [len(x) for x in test_data]
    
    print(f"📊 Average review length (before padding):")
    print(f"   Training: {np.mean(train_lengths):.1f} words")
    print(f"   Testing: {np.mean(test_lengths):.1f} words")
    print(f"📏 After padding/truncating: {sequence_length} words")
    print(f"✅ Preprocessing complete!")
    
    return train_data_padded, test_data_padded


def get_word_index():
    """
    This gets the dictionary that translates numbers back to words
    Like having a code book to decode secret messages! 🔐
    
    Returns: A dictionary mapping word IDs to actual words
    """
    word_index = imdb.get_word_index()
    
    # The dataset has special codes:
    # 0 = padding (blank space)
    # 1 = start of review
    # 2 = unknown word
    # So we shift all word IDs by 3
    word_index = {k: (v + 3) for k, v in word_index.items()}
    word_index["<PAD>"] = 0
    word_index["<START>"] = 1
    word_index["<UNK>"] = 2
    word_index["<UNUSED>"] = 3
    
    # Create reverse index (number -> word)
    reverse_word_index = {value: key for key, value in word_index.items()}
    
    return word_index, reverse_word_index


def decode_review(encoded_review, reverse_word_index):
    """
    This translates number codes back into words we can read!
    Like translating from robot language to human language 🤖➡️👨
    
    encoded_review: List of numbers representing words
    reverse_word_index: The translation dictionary
    
    Returns: The actual text of the review
    """
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])


def print_sample_reviews(train_data, train_labels, num_samples=3):
    """
    This shows us a few example reviews so we can see what we're working with!
    Like peeking at a few pages of a book 📖
    
    train_data: The training reviews
    train_labels: Whether each review is positive (1) or negative (0)
    num_samples: How many examples to show
    """
    print(f"\n📝 Here are {num_samples} example reviews:\n")
    
    # Get the word dictionary
    _, reverse_word_index = get_word_index()
    
    for i in range(num_samples):
        # Decode the review from numbers to words
        review_text = decode_review(train_data[i], reverse_word_index)
        sentiment = "😊 POSITIVE" if train_labels[i] == 1 else "😞 NEGATIVE"
        
        print(f"Review #{i+1} - {sentiment}")
        print(f"Length: {len(train_data[i])} words")
        print(f"Text: {review_text[:200]}...")  # Show first 200 characters
        print("-" * 80)
        print()


if __name__ == "__main__":
    """
    This runs when you execute this file directly
    It's like a test to make sure everything works! 🧪
    """
    print("🧪 Testing the preprocessing functions...\n")
    
    # Load the data
    train_data, train_labels, test_data, test_labels = load_imdb_data()
    
    # Show some example reviews
    print_sample_reviews(train_data, train_labels)
    
    # Preprocess with different sequence lengths
    for seq_len in [25, 50, 100]:
        print(f"\n{'='*60}")
        train_padded, test_padded = preprocess_data(train_data, test_data, sequence_length=seq_len)
        print(f"Train shape: {train_padded.shape}")
        print(f"Test shape: {test_padded.shape}")
    
    print("\n✅ All preprocessing tests passed!")