# download_data.py
# This is a helper robot that downloads movie reviews for us!
# Think of it like a librarian bringing us books 📚

import tensorflow as tf
from tensorflow.keras.datasets import imdb
import numpy as np
import os

print("🎬 Starting to download movie reviews...")
print("This might take a few minutes, like waiting for popcorn! 🍿")

# Download the IMDb dataset
# It has 50,000 movie reviews - people saying if movies are good or bad!
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(f"✅ Downloaded successfully!")
print(f"📊 Training reviews: {len(train_data)}")
print(f"📊 Testing reviews: {len(test_data)}")

# Save the data in the 'data' folder
print("💾 Saving data to the 'data' folder...")

# Create data folder if it doesn't exist
os.makedirs('data', exist_ok=True)

# Save everything
np.save('data/train_data.npy', train_data)
np.save('data/train_labels.npy', train_labels)
np.save('data/test_data.npy', test_data)
np.save('data/test_labels.npy', test_labels)

print("🎉 All done! Data is saved and ready!")
print("✨ You're doing AMAZING! Ready for the next step!")