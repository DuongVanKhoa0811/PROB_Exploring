import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import pmath


# Utils function
class ToPoincare(nn.Module):
    r"""
    Module which maps points in n-dim Euclidean space
    to n-dim Poincare ball
    Also implements clipping from https://arxiv.org/pdf/2107.11472.pdf
    """

    def __init__(self, c, train_c=False, train_x=False, ball_dim=None, riemannian=True, clip_r=None):
        super(ToPoincare, self).__init__()
        if train_x:
            if ball_dim is None:
                raise ValueError(
                    "if train_x=True, ball_dim has to be integer, got {}".format(
                        ball_dim
                    )
                )
            self.xp = nn.Parameter(torch.zeros((ball_dim,)))
        else:
            self.register_parameter("xp", None)

        if train_c:
            self.c = nn.Parameter(torch.Tensor([c,]))
        else:
            self.c = c

        self.train_x = train_x

        self.riemannian = pmath.RiemannianGradient
        self.riemannian.c = c
        
        self.clip_r = clip_r
        
        if riemannian:
            self.grad_fix = lambda x: self.riemannian.apply(x)
        else:
            self.grad_fix = lambda x: x

    def forward(self, x):
        if self.clip_r is not None:
            #ForkedPdb().set_trace()
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac =  torch.minimum(
                torch.ones_like(x_norm), 
                self.clip_r / x_norm
            )
            x = x * fac
            
        if self.train_x:
            xp = pmath.project(pmath.expmap0(self.xp, c=self.c), c=self.c)
            return self.grad_fix(pmath.project(pmath.expmap(xp, x, c=self.c), c=self.c))
        return self.grad_fix(pmath.project(pmath.expmap0(x, c=self.c), c=self.c))

    def extra_repr(self):
        return "c={}, train_x={}".format(self.c, self.train_x)
    
def load_obj_features():
    with h5py.File('/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/objfeatures_V10.h5', 'r') as file:
        with h5py.File('/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/objfeatures_V10_class_name.h5', 'r') as class_name_file:
            print(len(file.keys()))
            layers_obj_features = {}
            layers_obj_features_class_name = []
            
            # Load data
            for idx, sample_key in enumerate(file.keys()):
                class_names = class_name_file[sample_key][:]
                class_names = [name.decode('utf-8') for name in class_names]
                layers_obj_features_class_name.extend(class_names)
                
                for key in file[sample_key].keys():
                    for subkey in file[sample_key][key].keys():
                        if subkey not in layers_obj_features:
                            layers_obj_features[subkey] = []
                        layers_obj_features[subkey].append(np.array(file[sample_key][key][subkey]))

            # Post processing            
            for subkey in layers_obj_features.keys():
                layers_obj_features[subkey] = np.concatenate(layers_obj_features[subkey], axis=0)
            layers_obj_features_class_name = np.array(layers_obj_features_class_name)

    return layers_obj_features, layers_obj_features_class_name

def to_hyperbolic(layers_obj_features):
    
    tpc = ToPoincare(c=0.1,ball_dim=256,riemannian=False,clip_r=1.0)
    
    for subkey, obj_features in layers_obj_features.items():
        print('Processing subkey: ', subkey)
        layers_obj_features[subkey] = tpc(torch.from_numpy(obj_features).unsqueeze(0))
        layers_obj_features[subkey] = layers_obj_features[subkey].squeeze(0).numpy()
        print('Processed subkey: ', subkey, layers_obj_features[subkey].shape)
    
    return layers_obj_features


if __name__ == '__main__':
    # TSNE Visualization
    def tsne_visualization(layers_obj_features, layers_obj_features_class_name):

        # Get unique classes and create a mapping to numeric labels
        unique_classes = np.unique(layers_obj_features_class_name)
        class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
        print(f"Number of unique classes: {len(unique_classes)}")

        for subkey, obj_features in layers_obj_features.items():
            print(subkey, layers_obj_features[subkey].shape)

            data = layers_obj_features[subkey]
            labels = layers_obj_features_class_name  # class names for all points
            
            m = 5000  # number of samples you want
            indices = np.random.choice(data.shape[0], m, replace=False)
            sampled_data = data[indices]  # shape: (m, 256)
            sampled_labels = labels[indices]  # Sample corresponding labels too
            
            # Convert class names to numeric for coloring
            numeric_labels = np.array([class_to_idx[lbl] for lbl in sampled_labels])

            # Apply t-SNE to reduce to 2D
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            data_2d = tsne.fit_transform(sampled_data)

            # Plot with class-based colors
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                                c=numeric_labels, 
                                cmap='tab20',  # Use 'tab20' for up to 20 classes, or 'nipy_spectral' for more
                                alpha=0.6, 
                                s=10)
            
            # Add colorbar or legend
            cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
            cbar.ax.set_yticklabels(unique_classes)
            cbar.set_label('Class')
            
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title(f't-SNE Visualization - {subkey}')
            plt.tight_layout()
            plt.savefig(f'../trash/tsne_plot_{subkey}.png', dpi=150)
            plt.close()

    # layers_obj_features, layers_obj_features_class_name = load_obj_features()
    # tsne_visualization(layers_obj_features, layers_obj_features_class_name)


    # Hyperbolic visualization
    layers_obj_features, layers_obj_features_class_name = load_obj_features()
    hyperbolic_features = to_hyperbolic(layers_obj_features)
    tsne_visualization(hyperbolic_features, layers_obj_features_class_name)
    