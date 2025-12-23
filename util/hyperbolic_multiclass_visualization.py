# Github Link for this code: https://github.com/geoopt/geoopt/blob/master/examples/hyperbolic_multiclass_classification.ipynb

import matplotlib.pyplot as plt
import geoopt
import torch
import itertools
import torch.nn as nn
import numpy as np
import tqdm


#@title
class SyntheticDataset(torch.utils.data.Dataset):
    '''
    Adopted from https://github.com/emilemathieu/pvae/blob/ca5c4997a90839fc8960ec812df4cbf83da55781/pvae/datasets/datasets.py
    
    Implementation of a synthetic dataset by hierarchical diffusion. 
    Args:
    :param int dim: dimension of the input sample
    :param int depth: depth of the tree; the root corresponds to the depth 0
    :param int :numberOfChildren: Number of children of each node in the tree
    :param int :numberOfsiblings: Number of noisy observations obtained from the nodes of the tree
    :param float sigma_children: noise
    :param int param: integer by which :math:`\\sigma_children` is divided at each deeper level of the tree
    '''
    def __init__(self, ball, dim, depth, numberOfChildren=2, dist_children=1, sigma_sibling=2, param=1, numberOfsiblings=1):
        assert numberOfChildren == 2
        self.dim = int(dim)
        self.ball = ball
        self.root = ball.origin(self.dim)
        self.sigma_sibling = sigma_sibling
        self.depth = int(depth)
        self.dist_children = dist_children
        self.numberOfChildren = int(numberOfChildren)
        self.numberOfsiblings = int(numberOfsiblings)  
        self.__class_counter = itertools.count()
        self.origin_data, self.origin_labels, self.data, self.labels = map(torch.detach, self.bst())
        self.num_classes = self.origin_labels.max().item()+1
        #self.data = ball.mobius_add(self.data, -ball.weighted_midpoint(self.data)).detach()

    def __len__(self):
        '''
        this method returns the total number of samples/nodes
        '''
        return len(self.data)

    def __getitem__(self, idx):
        '''
        Generates one sample
        '''
        data, labels = self.data[idx], self.labels[idx]
        return data, labels, labels.max(-1).values

    def get_children(self, parent_value, parent_label, current_depth, offspring=True):
        '''
        :param 1d-array parent_value
        :param 1d-array parent_label
        :param int current_depth
        :param  Boolean offspring: if True the parent node gives birth to numberOfChildren nodes
                                    if False the parent node gives birth to numberOfsiblings noisy observations
        :return: list of 2-tuples containing the value and label of each child of a parent node
        :rtype: list of length numberOfChildren
        '''
        if offspring:
            numberOfChildren = self.numberOfChildren
            sigma = self.dist_children
        else:
            numberOfChildren = self.numberOfsiblings
            sigma = self.sigma_sibling
        if offspring:
            direction = torch.randn_like(parent_value)
            parent_value_n = parent_value / parent_value.norm().clamp_min(1e-15)
            direction -= parent_value_n @ direction * parent_value_n
            child_value_1 = ball.geodesic_unit(torch.tensor(sigma), parent_value, direction)
            child_value_2 = ball.geodesic_unit(torch.tensor(sigma), parent_value, -direction)
            child_label_1 = parent_label.clone()
            child_label_1[current_depth] = next(self.__class_counter)
            child_label_2 = parent_label.clone()
            child_label_2[current_depth] = next(self.__class_counter)
            children = [
                (child_value_1, child_label_1),
                (child_value_2, child_label_2)
            ]
        else:
            children = []
            for i in range (numberOfChildren):
                child_value = ball.random(self.dim, mean=parent_value, std=sigma ** .5)
                child_label = parent_label.clone()
                children.append((child_value, child_label))
        return children

    def bst(self):
        '''
        This method generates all the nodes of a level before going to the next level
        '''
        label = -torch.ones(self.depth+1, dtype=torch.long)
        label[0] = next(self.__class_counter)
        queue = [(self.root, label, 0)]
        visited = []
        labels_visited = []
        values_clones = []
        labels_clones = []
        while len(queue) > 0:
            current_node, current_label, current_depth = queue.pop(0)
            visited.append(current_node)
            labels_visited.append(current_label)
            if current_depth < self.depth:
                children = self.get_children(current_node, current_label, current_depth)
                for child in children:
                    queue.append((child[0], child[1], current_depth + 1)) 
            if current_depth <= self.depth:
                clones = self.get_children(current_node, current_label, current_depth, False)
                for clone in clones:
                    values_clones.append(clone[0])
                    labels_clones.append(clone[1])
        length = int(((self.numberOfChildren) ** (self.depth + 1) - 1) / (self.numberOfChildren - 1))
        length_leaves = int(self.numberOfChildren**self.depth)
        images = torch.cat([i for i in visited]).reshape(length, self.dim)
        labels_visited = torch.cat([i for i in labels_visited]).reshape(length, self.depth+1)[:,:self.depth]
        values_clones = torch.cat([i for i in values_clones]).reshape(self.numberOfsiblings*length, self.dim)
        labels_clones = torch.cat([i for i in labels_clones]).reshape(self.numberOfsiblings*length, self.depth+1)
        return images, labels_visited, values_clones, labels_clones
    
ball = geoopt.PoincareBall()
torch.manual_seed(42)
dataset = SyntheticDataset(ball, 2, 3, numberOfsiblings=100, sigma_sibling=0.02, dist_children=1)
num_classes = dataset.num_classes


#@title long util funtion to plot geodesic grid
def add_geodesic_grid(ax: plt.Axes, manifold: geoopt.Stereographic, line_width=0.1):

    # define geodesic grid parameters
    N_EVALS_PER_GEODESIC = 10000
    STYLE = "--"
    COLOR = "gray"
    LINE_WIDTH = line_width

    # get manifold properties
    K = manifold.k.item()
    R = manifold.radius.item()

    # get maximal numerical distance to origin on manifold
    if K < 0:
        # create point on R
        r = torch.tensor((R, 0.0), dtype=manifold.dtype)
        # project point on R into valid range (epsilon border)
        r = manifold.projx(r)
        # determine distance from origin
        max_dist_0 = manifold.dist0(r).item()
    else:
        max_dist_0 = np.pi * R
    # adjust line interval for spherical geometry
    circumference = 2*np.pi*R

    # determine reasonable number of geodesics
    # choose the grid interval size always as if we'd be in spherical
    # geometry, such that the grid interpolates smoothly and evenly
    # divides the sphere circumference
    n_geodesics_per_circumference = 4 * 6  # multiple of 4!
    n_geodesics_per_quadrant = n_geodesics_per_circumference // 2
    grid_interval_size = circumference / n_geodesics_per_circumference
    if K < 0:
        n_geodesics_per_quadrant = int(max_dist_0 / grid_interval_size)

    # create time evaluation array for geodesics
    if K < 0:
        min_t = -1.2*max_dist_0
    else:
        min_t = -circumference/2.0
    t = torch.linspace(min_t, -min_t, N_EVALS_PER_GEODESIC)[:, None]

    # define a function to plot the geodesics
    def plot_geodesic(gv):
        ax.plot(*gv.t().numpy(), STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # define geodesic directions
    u_x = torch.tensor((0.0, 1.0))
    u_y = torch.tensor((1.0, 0.0))

    # add origin x/y-crosshair
    o = torch.tensor((0.0, 0.0))
    if K < 0:
        x_geodesic = manifold.geodesic_unit(t, o, u_x)
        y_geodesic = manifold.geodesic_unit(t, o, u_y)
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
    else:
        # add the crosshair manually for the sproj of sphere
        # because the lines tend to get thicker if plotted
        # as done for K<0
        ax.axvline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)
        ax.axhline(0, linestyle=STYLE, color=COLOR, linewidth=LINE_WIDTH)

    # add geodesics per quadrant
    for i in range(1, n_geodesics_per_quadrant):
        i = torch.as_tensor(float(i))
        # determine start of geodesic on x/y-crosshair
        x = manifold.geodesic_unit(i*grid_interval_size, o, u_y)
        y = manifold.geodesic_unit(i*grid_interval_size, o, u_x)

        # compute point on geodesics
        x_geodesic = manifold.geodesic_unit(t, x, u_x)
        y_geodesic = manifold.geodesic_unit(t, y, u_y)

        # plot geodesics
        plot_geodesic(x_geodesic)
        plot_geodesic(y_geodesic)
        if K < 0:
            plot_geodesic(-x_geodesic)
            plot_geodesic(-y_geodesic)
            
            
plt.figure(figsize=(10, 10))
circle = plt.Circle((0, 0), 1, fill=False, color="b")
add_geodesic_grid(plt.gca(), ball, 0.5)
plt.gca().add_artist(circle)
plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)
plt.gca().set_aspect("equal")
plt.scatter(*dataset.data.T, 
            c=dataset.labels.max(-1).values.float() / num_classes, 
            alpha=.3, cmap="rainbow");
plt.savefig('../trash/hyperbolic_multiclass_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
