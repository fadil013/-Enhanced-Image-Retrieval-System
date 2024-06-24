import numpy as np
import cv2
import os
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI plotting
import matplotlib.pyplot as plt

class LSH:
    def __init__(self, num_tables, num_functions, num_buckets, num_dimensions):
        self.num_tables = num_tables
        self.num_functions = num_functions
        self.num_buckets = num_buckets
        self.num_dimensions = num_dimensions
        self.projections = [self.generate_projection_matrix() for _ in range(self.num_tables)]
        self.tables = [{} for _ in range(self.num_tables)]

    def generate_projection_matrix(self):
        return np.random.randn(self.num_functions, self.num_dimensions)

    def hash_function(self, vec, proj_matrix):
        return tuple((np.dot(proj_matrix, vec) > 0).astype(int))

    def hash_vector(self, vec):
        for i in range(self.num_tables):
            hash_value = self.hash_function(vec, self.projections[i])
            hash_key = tuple(hash_value)
            table_index = hash(hash_key) % self.num_buckets
            if table_index in self.tables[i]:
                self.tables[i][table_index].append(tuple(vec.tolist()))  # Convert numpy array to tuple
            else:
                self.tables[i][table_index] = [tuple(vec.tolist())]  # Convert numpy array to tuple if key doesn't exist

    def query(self, query_vec):
        results = set()
        for i in range(self.num_tables):
            hash_value = self.hash_function(query_vec, self.projections[i])
            hash_key = tuple(hash_value)
            table_index = hash(hash_key) % self.num_buckets
            if table_index in self.tables[i]:
                results.update(self.tables[i][table_index])
        return results

# Function to compute Euclidean distance between two vectors
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# Load CIFAR-10 dataset
def load_cifar10_dataset(root_dir, train=True):
    images = []
    labels = []
    sub_dir = 'train' if train else 'test'
    dataset_dir = os.path.join(root_dir)
    for file in os.listdir(dataset_dir):
        if file.endswith(".png"):
            label_str = file[:-4]  # Remove the ".png" extension
            try:
                label = int(label_str)
            except ValueError:
                print("Invalid label for file:", file)
                continue
            image_path = os.path.join(dataset_dir, file)
            image = cv2.imread(image_path)
            images.append(image)
            labels.append(label)
            #print("Loaded image:", image_path)  # Print path of loaded image
    print("Number of images loaded:", len(images))  # Print number of loaded images
    return images, labels

# Pre-process and extract features
def preprocess_and_extract_features(images):
    features = []
    for idx, image in enumerate(images):
        # Resize image to a smaller size (e.g., 32x32 pixels)
        resized_image = cv2.resize(image, (32, 32))
        
        # Convert image to grayscale
        gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        
        # Compute histogram for grayscale image
        gray_hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256]).flatten()
        
        # Perform edge detection using Canny edge filter
        edges = cv2.Canny(resized_image, 100, 200)
        
        # Visualize grayscale image and edges for the first 5 images only
        if idx < 5:
            plt.ioff()  # Turn off interactive mode
            plt.figure(figsize=(10, 6))
            plt.subplot(231)
            plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
            plt.title('Original Image')
            plt.subplot(232)
            plt.imshow(gray_image, cmap='gray')
            plt.title('Grayscale Image')
            plt.subplot(233)
            plt.imshow(edges, cmap='gray')
            plt.title('Edges (Canny)')
            
            # Compute color histograms
            color_hist = []
            for channel in cv2.split(resized_image):
                hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).flatten()
                color_hist.append(hist)
            color_hist = np.concatenate(color_hist)
            
            # Visualize color histograms
            plt.subplot(234)
            plt.plot(color_hist[:256], color='r')
            plt.title('Red Channel Histogram')
            plt.subplot(235)
            plt.plot(color_hist[256:512], color='g')
            plt.title('Green Channel Histogram')
            plt.subplot(236)
            plt.plot(color_hist[512:], color='b')
            plt.title('Blue Channel Histogram')
            
            plt.tight_layout()
            plt.show()
        
        # Concatenate histograms and edge information to form feature vector
        hist_features = np.concatenate([gray_hist, edges.flatten(), color_hist])
        features.append(hist_features)
    
    return np.array(features)

# Specify the root directory where CIFAR-10 dataset is stored
root_dir = r"C:\Users\HTS\Desktop\i221925-i221947-i221815_AbdullahNadeem-HarrisHassan-FadilAwan_A1"


# Load CIFAR-10 training dataset
train_images, train_labels = load_cifar10_dataset(root_dir, train=True)

# Pre-process and extract features from training images
X_train = preprocess_and_extract_features(train_images)

# Instantiate LSH object
num_tables = 5
num_functions = 10
num_buckets = 1000
num_dimensions = X_train.shape[1]  # Dimensionality of feature vectors
lsh = LSH(num_tables, num_functions, num_buckets, num_dimensions)

# Populate LSH tables with training data
for vec in X_train:
    lsh.hash_vector(vec)

# Select a subset of images from the training set as queries
num_queries = min(10, len(train_images))  # Ensure num_queries does not exceed number of available images
query_indices = np.random.choice(len(train_images), num_queries, replace=False)
query_images = [train_images[idx] for idx in query_indices]
query_features = [X_train[idx] for idx in query_indices]

# Execute queries against LSH tables
for i, query_feature in enumerate(query_features):
    neighbors = lsh.query(query_feature)
    print(f"Query {i+1}: Found {len(neighbors)} neighbor(s) using LSH.")
    
    # Compute distances between query vector and retrieved neighbors
    distances = [euclidean_distance(query_feature, neighbor) for neighbor in neighbors]
    
    # Sort neighbors by distance
    sorted_neighbors = sorted(zip(neighbors, distances), key=lambda x: x[1])
    
    # Print top 5 nearest neighbors using Euclidean distance
    print(f"Query {i+1}: Nearest neighbors using Euclidean distance (LSH):")
    for j, (neighbor, distance) in enumerate(sorted_neighbors[:5]):
        print(f"Neighbor {j+1}: Euclidean distance = {distance}")

    # Compute cosine similarity between query vector and retrieved neighbors
    similarities = [cosine_similarity(query_feature, neighbor) for neighbor in neighbors]
    
    # Sort neighbors by similarity
    sorted_neighbors_cosine = sorted(zip(neighbors, similarities), key=lambda x: x[1], reverse=True)
    
    # Print top 5 most similar neighbors using cosine similarity
    print(f"Query {i+1}: Nearest neighbors using Cosine similarity (LSH):")
    for j, (neighbor, similarity) in enumerate(sorted_neighbors_cosine[:5]):
        print(f"Neighbor {j+1}: Cosine similarity = {similarity}")
