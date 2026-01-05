import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os

# Add the models directory to path to import pmath
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import pmath

# Import the load_obj_features function
from models.prob_features_visualization import load_obj_features


def calculate_euclidean_distance(x, y):
    """Calculate L2 (Euclidean) distance between two feature vectors."""
    return np.linalg.norm(x - y)


def calculate_hyperbolic_distance(x, y, c=0.1):
    """Calculate hyperbolic distance using pmath.dist."""
    x_tensor = torch.from_numpy(x).float()
    y_tensor = torch.from_numpy(y).float()
    dist = pmath.dist(x_tensor, y_tensor, c=c)
    return dist.item()


def sample_object_pairs(n_pairs, num_objects, random_seed=42):
    """
    Sample n_pairs of object indices using direct sampling.
    
    Args:
        n_pairs: Number of pairs to sample
        num_objects: Total number of objects
        random_seed: Random seed for reproducibility
    
    Returns:
        List of (i, j) tuples representing object pairs
    """
    np.random.seed(random_seed)
    max_possible_pairs = num_objects * (num_objects - 1) // 2
    
    # If we need more pairs than possible, use all pairs
    if n_pairs >= max_possible_pairs:
        pairs = [(i, j) for i in range(num_objects) for j in range(i+1, num_objects)]
        return pairs
    
    # Use a set to track unique pairs (faster lookup)
    seen_pairs = set()
    pairs = []
    
    while len(pairs) < n_pairs:
        # Sample two distinct indices
        i, j = np.random.choice(num_objects, size=2, replace=False)
        
        # Ensure i < j for consistency
        if i > j:
            i, j = j, i
        
        pair = (i, j)
        if pair not in seen_pairs:
            pairs.append(pair)
            seen_pairs.add(pair)
    
    return pairs


def analyze_distance_distributions(layers_obj_features, hyperbolic_layer, euclidean_layer, n_pairs=1000, c=0.1, random_seed=42):
    """
    Analyze distance distributions for Euclidean and hyperbolic spaces.
    
    Args:
        layers_obj_features: Dictionary with layer names as keys and feature arrays as values
        hyperbolic_layer: Name of hyperbolic layer
        euclidean_layer: Name of Euclidean layer
        n_pairs: Number of object pairs to sample for distance calculation
        c: Curvature parameter for hyperbolic space (default: 0.1)
        random_seed: Random seed for reproducibility
    """
    
    obj_temp = 1.3
    hidden_dim = 256
    
    def PROB_obj_score_calculation(distances, temperature):
        """
        Calculate objectness scores for PROB.
        
        Args:
            distances: List of distances
        """
        
        return [np.exp(-temperature * distance ** 2) for distance in distances]

    # Identify hyperbolic layer
    print(f"Found 1 Euclidean layer and 1 hyperbolic layer")
    print(f"Euclidean layers: {euclidean_layer}")
    print(f"Hyperbolic layer: {hyperbolic_layer}")
    
    # Get number of objects (assuming all layers have same number of objects)
    num_objects = layers_obj_features[list(layers_obj_features.keys())[0]].shape[0]
    print(f"Total number of objects: {num_objects}")
    
    # Sample object pairs
    pairs = sample_object_pairs(n_pairs, num_objects, random_seed)
    print(f"Sampled {len(pairs)} object pairs")
    
    # Calculate distances for each layer
    layer_distances = {}
    layer_objectness = {}

    # Calculate distances for Euclidean layer
    features = layers_obj_features[euclidean_layer]  # Shape: (N_object, dim)
    mean = features.mean(axis=0)  # Shape: (dim,)
    std = features.std(axis=0, ddof=0)  # Shape: (dim,), ddof=0 for population std (unbiased=False)
    eps = 1e-5  # Default eps for BatchNorm
    normalized_features = (features - mean) / (std + eps)
    distances = []
    for i, j in pairs:
        dist = calculate_euclidean_distance(normalized_features[i], normalized_features[j])
        distances.append(dist)
    layer_distances[euclidean_layer] = np.array(distances)
    print(f"{euclidean_layer} (Euclidean, normalized): mean={np.mean(distances):.4f}, std={np.std(distances):.4f}")
    
    # Calculate objectness scores for Euclidean layer
    temperature=obj_temp/hidden_dim
    objectness_scores = PROB_obj_score_calculation(distances, temperature)
    layer_objectness[euclidean_layer] = objectness_scores
    print(f"{euclidean_layer} (Objectness): mean={np.mean(objectness_scores):.4f}, std={np.std(objectness_scores):.4f}")
    
    # Calculate distances for hyperbolic layer
    assert hyperbolic_layer in layers_obj_features
    features = layers_obj_features[hyperbolic_layer]
    distances = []
    for i, j in pairs:
        dist = calculate_hyperbolic_distance(features[i], features[j], c=c)
        distances.append(dist)
    layer_distances[hyperbolic_layer] = np.array(distances)
    print(f"{hyperbolic_layer} (Hyperbolic): mean={np.mean(distances):.4f}, std={np.std(distances):.4f}")
    
    # Calculate objectness scores for hyperbolic layer
    for i in range(1,151,10):
        temperature= i * obj_temp/hidden_dim
        objectness_scores = PROB_obj_score_calculation(distances, temperature)
        layer_objectness[hyperbolic_layer] = objectness_scores
        print(f"{hyperbolic_layer} (Objectness): mean={np.mean(objectness_scores):.4f}, std={np.std(objectness_scores):.4f}")
        
        # Plot distance distributions
        plot_distance_distributions(layer_distances, layer_objectness, euclidean_layer, hyperbolic_layer, n_pairs, suffix_name=f'_{i}_times_{temperature}')
    
    return layer_distances, layer_objectness


def plot_distance_distributions(layer_distances, layer_objectness, euclidean_layer, hyperbolic_layer, n_pairs, suffix_name=''):
    """
    Plot distance distributions and objectness scores for all layers.
    
    Args:
        layer_distances: Dictionary with layer names and their distance arrays
        layer_objectness: Dictionary with layer names and their objectness scores
        euclidean_layer: Name of Euclidean layer
        hyperbolic_layer: Name of hyperbolic layer
        n_pairs: Number of pairs used (for title)
    """
    label_fontsize = 15
    title_fontsize = 18
    tick_fontsize = 15

    # Create a comparison plot: Euclidean vs Hyperbolic (distances)
    fig, ax = plt.subplots(figsize=(14, 8))
    distances_euc = layer_distances[euclidean_layer]
    distances_hyp = layer_distances[hyperbolic_layer]
    
    # Calculate mean and std
    mean_euc = np.mean(distances_euc)
    std_euc = np.std(distances_euc)
    mean_hyp = np.mean(distances_hyp)
    std_hyp = np.std(distances_hyp)
    
    ax.hist(distances_euc, bins=50, alpha=0.5, 
            label=f'{euclidean_layer} (Euclidean)\nMean: {mean_euc:.4f}, Std: {std_euc:.4f}', 
            edgecolor='black', color='lightblue')
    ax.hist(distances_hyp, bins=50, alpha=0.7, 
            label=f'{hyperbolic_layer} (Hyperbolic)\nMean: {mean_hyp:.4f}, Std: {std_hyp:.4f}', 
            edgecolor='black', color='orange')
    ax.set_xlabel('Distance', fontsize=label_fontsize)
    ax.set_ylabel('Frequency', fontsize=label_fontsize)
    ax.set_title(f'Distance Distribution Comparison: Euclidean vs Hyperbolic\n({n_pairs} object pairs)', 
                 fontsize=title_fontsize, fontweight='bold')
    ax.legend(fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../trash/distance_comparison_euclidean_vs_hyperbolic.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot to '../trash/distance_comparison_euclidean_vs_hyperbolic.png'")
    plt.close()
    
    # Create a comparison plot: Euclidean vs Hyperbolic (objectness scores)
    fig, ax = plt.subplots(figsize=(12, 6))
    objectness_euc = layer_objectness[euclidean_layer]
    objectness_hyp = layer_objectness[hyperbolic_layer]
    ax.hist(objectness_euc, bins=50, alpha=0.5, label=f'{euclidean_layer} (Euclidean)', 
            edgecolor='black', color='lightblue')
    ax.hist(objectness_hyp, bins=50, alpha=0.7, label=f'{hyperbolic_layer} (Hyperbolic)', 
            edgecolor='black', color='orange')
    ax.set_xlabel('Objectness Score', fontsize=label_fontsize)
    ax.set_ylabel('Frequency', fontsize=label_fontsize)
    ax.set_title(f'Objectness Score Distribution Comparison: Euclidean vs Hyperbolic\n({n_pairs} object pairs)', 
                 fontsize=title_fontsize, fontweight='bold')
    ax.legend(fontsize=label_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'../trash/objectness_comparison_euclidean_vs_hyperbolic{suffix_name}.png', dpi=150, bbox_inches='tight')
    print(f"Saved objectness comparison plot to '../trash/objectness_comparison_euclidean_vs_hyperbolic{suffix_name}.png'")
    plt.close()


if __name__ == '__main__':
    # Configuration
    objfeatures_filename = 'objfeatures_V18_IoU06'
    background_features_filename = 'objfeatures_V18_less_IoU01'
    n_pairs = 5000  # Number of object pairs to sample
    c = 0.1  # Curvature parameter for hyperbolic space
    hyperbolic_layer = 'tpc.0_out'
    euclidean_layer = 'feature_projector.0.norm_out'
    
    print("Loading object features...")
    layers_obj_features, layers_obj_features_class_name = load_obj_features(
        objfeatures_filename, 
        background_features_filename
    )
    
    print(f"\nLoaded features for {len(layers_obj_features)} layers")
    print(f"Layer names: {list(layers_obj_features.keys())}")
    
    print(f"\nAnalyzing distance distributions with {n_pairs} object pairs...")
    layer_distances = analyze_distance_distributions(
        {key:value for key, value in layers_obj_features.items() if '_out' in key and 'transformer.decoder.layers.5.norm3_out' != key}, 
        n_pairs=n_pairs, 
        c=c, 
        random_seed=42,
        hyperbolic_layer=hyperbolic_layer,
        euclidean_layer=euclidean_layer
    )
    
    print("\nAnalysis complete!")
