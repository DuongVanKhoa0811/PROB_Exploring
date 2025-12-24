import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Circle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import pmath


if __name__ == '__main__':
    
    # Choose visualization method: 'pca' or 'tsne'
    method = 'pca'  # Change to 'tsne' to use t-SNE instead
    
    hidden_dim = 256
    n_query = 100
    c = 0.1
    boundary_radius = 1.0 / (c ** 0.5)  # ≈ 3.16
    
    # Generate points at specific fractions of boundary radius
    _range = torch.linspace(0.1, 1, 10)  # 10% to 99% of boundary
    
    obj_centroid = torch.zeros(n_query, hidden_dim)
    
    # Collect all features and labels
    all_features = []
    all_labels = []
    all_fractions = []
    
    for fraction in _range:
        # Generate random unit vectors
        random_directions = torch.randn(n_query, hidden_dim)
        random_directions = random_directions / (torch.norm(random_directions, dim=-1, keepdim=True) + 1e-8)
        
        # Scale to specific fraction of boundary
        target_radius = fraction * boundary_radius
        points = random_directions * target_radius
        
        # Project to ensure they're on the ball
        features = pmath.project(points, c=c)
        
        dists = pmath.dist(features, obj_centroid, c=c)
        actual_norms = torch.norm(features, dim=-1)
        
        print(f"Fraction: {fraction:.5f}, Target: {target_radius:.5f}, "
              f"Actual norm: {actual_norms.mean():.5f}±{actual_norms.std():.5f}, "
              f"Distance: {dists.mean():.5f}±{dists.std():.5f}, "
              f"Max dist: {dists.max():.5f}")
        
        # Collect features
        all_features.append(features.cpu().numpy())
        all_labels.extend([f'Fraction {fraction:.2f}'] * n_query)
        all_fractions.extend([fraction.item()] * n_query)
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)  # [n_query * len(_range), hidden_dim]
    
    print(f"\nTotal features: {all_features.shape}")
    
    # Apply dimensionality reduction
    if method == 'pca':
        print("Computing PCA...")
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(all_features)
        
        # Project PCA results to Poincaré disk (normalize to unit circle)
        max_norm = np.linalg.norm(features_2d, axis=1).max()
        if max_norm > 0:
            features_2d_normalized = features_2d / (max_norm * 1.1)  # Scale to fit within unit circle
        else:
            features_2d_normalized = features_2d
        
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"Total variance explained: {reducer.explained_variance_ratio_.sum():.4f}")
        
        xlabel = 'PCA Component 1'
        ylabel = 'PCA Component 2'
        title_suffix = f'PCA)\nExplained Variance: {reducer.explained_variance_ratio_.sum():.2%}'
        save_filename = '../trash/pca_poincare_points.png'
        show_unit_circle = True
        
    elif method == 'tsne':
        print("Computing t-SNE...")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
        features_2d = reducer.fit_transform(all_features)
        features_2d_normalized = features_2d  # t-SNE doesn't need normalization
        
        xlabel = 't-SNE 1'
        ylabel = 't-SNE 2'
        title_suffix = 't-SNE)'
        save_filename = '../trash/tsne_poincare_points.png'
        show_unit_circle = False  # t-SNE doesn't preserve Poincaré disk structure
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'pca' or 'tsne'")
    
    # Create visualization
    figsize = (10, 10) if method == 'pca' else (12, 10)
    plt.figure(figsize=figsize)
    
    # Draw unit circle for PCA (Poincaré disk boundary)
    if show_unit_circle:
        circle = plt.Circle((0, 0), 1, fill=False, color="b", linewidth=2)
        plt.gca().add_artist(circle)
    
    # Get unique fractions for coloring
    unique_fractions = sorted(set(all_fractions))
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_fractions)))
    
    # Plot each fraction group
    for i, fraction in enumerate(unique_fractions):
        mask = np.array(all_fractions) == fraction
        plt.scatter(features_2d_normalized[mask, 0], features_2d_normalized[mask, 1], 
                   c=[colors[i]], label=f'Fraction {fraction:.2f}',
                   alpha=0.6, s=20)
    
    if show_unit_circle:
        plt.xlim(-1.1, 1.1)
        plt.ylim(-1.1, 1.1)
        plt.gca().set_aspect("equal")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f'Visualization of Points at Different Distances from Origin\n(Poincaré Ball, c=0.1, {title_suffix}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    if show_unit_circle:
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=150, bbox_inches='tight')
    print(f"Saved {method.upper()} plot to '{save_filename}'")
    plt.close()
    
    # Optional: Also plot distance vs fraction
    plt.figure(figsize=(10, 6))
    fractions_list = []
    distances_list = []
    
    for fraction in _range:
        random_directions = torch.randn(n_query, hidden_dim)
        random_directions = random_directions / (torch.norm(random_directions, dim=-1, keepdim=True) + 1e-8)
        target_radius = fraction * boundary_radius
        points = random_directions * target_radius
        features = pmath.project(points, c=c)
        dists = pmath.dist(features, obj_centroid, c=c)
        
        fractions_list.append(fraction.item())
        distances_list.append(dists.mean().item())
    
    plt.plot(fractions_list, distances_list, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Fraction of Boundary Radius')
    plt.ylabel('Mean Hyperbolic Distance')
    plt.title('Hyperbolic Distance vs Distance from Origin')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('../trash/distance_vs_fraction.png', dpi=150)
    print("Saved distance plot to '../trash/distance_vs_fraction.png'")
    plt.close()

