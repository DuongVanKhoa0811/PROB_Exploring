import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import pmath
import itertools


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

def to_hyperbolic(layers_obj_features):
    
    tpc = ToPoincare(c=0.1,ball_dim=256,riemannian=False,clip_r=1.0)
    
    for subkey, obj_features in layers_obj_features.items():
        print('Processing subkey: ', subkey)
        layers_obj_features[subkey] = tpc(torch.from_numpy(obj_features).unsqueeze(0))
        layers_obj_features[subkey] = layers_obj_features[subkey].squeeze(0).numpy()
        print('Processed subkey: ', subkey, layers_obj_features[subkey].shape)
    
    return layers_obj_features

def Hyp_OW_class_name():
    #OWOD splits
    VOC_CLASS_NAMES_COCOFIED = [
        "airplane",  "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]

    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant",  "sofa", "tvmonitor"
    ]
    UNK_CLASS = ["unknown"]

    VOC_COCO_CLASS_NAMES={}


    T1_CLASS_NAMES = [
        "aeroplane","bicycle","bird","boat","bus","car",
        "cat","cow","dog","horse","motorbike","sheep","train",
        "elephant","bear","zebra","giraffe","truck","person"
    ]

    T2_CLASS_NAMES = [
        "traffic light","fire hydrant","stop sign",
        "parking meter","bench","chair","diningtable",
        "pottedplant","backpack","umbrella","handbag",
        "tie","suitcase","microwave","oven","toaster","sink",
        "refrigerator","bed","toilet","sofa"
    ]

    T3_CLASS_NAMES = [
        "frisbee","skis","snowboard","sports ball",
        "kite","baseball bat","baseball glove","skateboard",
        "surfboard","tennis racket","banana","apple","sandwich",
        "orange","broccoli","carrot","hot dog","pizza","donut","cake"
    ]

    T4_CLASS_NAMES = [
        "laptop","mouse","remote","keyboard","cell phone","book",
        "clock","vase","scissors","teddy bear","hair drier","toothbrush",
        "wine glass","cup","fork","knife","spoon","bowl","tvmonitor","bottle"
    ]

    VOC_COCO_CLASS_NAMES["OWDETR"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))


    VOC_CLASS_NAMES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    VOC_CLASS_NAMES_COCOFIED = [
        "airplane",  "dining table", "motorcycle",
        "potted plant", "couch", "tv"
    ]

    BASE_VOC_CLASS_NAMES = [
        "aeroplane", "diningtable", "motorbike",
        "pottedplant",  "sofa", "tvmonitor"
    ]

    T2_CLASS_NAMES = [
        "truck", "traffic light", "fire hydrant", "stop sign", "parking meter",
        "bench", "elephant", "bear", "zebra", "giraffe",
        "backpack", "umbrella", "handbag", "tie", "suitcase",
        "microwave", "oven", "toaster", "sink", "refrigerator"
    ]

    T3_CLASS_NAMES = [
        "frisbee", "skis", "snowboard", "sports ball", "kite",
        "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake"
    ]

    T4_CLASS_NAMES = [
        "bed", "toilet", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "book", "clock",
        "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl"
    ]
    VOC_COCO_CLASS_NAMES["TOWOD"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))
    VOC_COCO_CLASS_NAMES["VOC2007"] = tuple(itertools.chain(VOC_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

    T1_CLASS_NAMES = [
        "bicycle","car","traffic light","fire hydrant",'bird','cat','dog',"backpack",
        'frisbee','skis', 'bottle','wine glass','banana','apple', 'chair','sofa','tvmonitor',
        'microwave','oven','book'
    ]

    T2_CLASS_NAMES = [
        'person','motorbike','aeroplane','stop sign',
    'horse','sheep','umbrella','snowboard','sports ball','cup','sandwich','orange','broccoli',
    'pottedplant','bed','laptop','mouse','toaster','clock','vase',
    ]

    T3_CLASS_NAMES = [
        'bus','train','parking meter','cow','elephant','bear','handbag','kite','baseball bat',
    'baseball glove'  ,'fork','knife','carrot','hot dog','diningtable','remote','keyboard',
    'sink','scissors','teddy bear'
    ]

    T4_CLASS_NAMES = [
        'truck','boat','bench','zebra','giraffe','tie','suitcase','skateboard','surfboard','tennis racket',
        'spoon','bowl','pizza','donut','cake','toilet','cell phone','refrigerator','hair drier',
    'toothbrush'

    
    ]
    # ForkedPdb().set_trace()

    VOC_COCO_CLASS_NAMES["HIERARCHICAL"] = tuple(itertools.chain(T1_CLASS_NAMES, T2_CLASS_NAMES, T3_CLASS_NAMES, T4_CLASS_NAMES, UNK_CLASS))

    return VOC_COCO_CLASS_NAMES

def Hyp_OW_Superclass_mapping(dataset, train_set):
      
    if 'HIERARCHICAL' in dataset:
        if 't1' in train_set:
            label_mapping={0:0,1:0,\
                                2:1,3:1,\
                                4:2,5:2,6:2,\
                                7:3,\
                                8:4  ,9:4,\
                                10:5, 11:5,\
                                12:6 , 13:6,\
                                14:7, 15:7,\
                                16:8,
                                17:9,18:9,
                                19:10
                                }
            family_mapping={0:torch.Tensor([0,1]).long(),\
                                1:torch.Tensor([2,3]).long(),  \
                                2:torch.Tensor([4,5,6]).long(),\
                                    3:torch.Tensor([7]).long(),\
                                    4:torch.Tensor([8,9]).long(),\
                                    5:torch.Tensor([10,11]).long(),\
                                    6:torch.Tensor([12,13]).long(),\
                                    7:torch.Tensor([14,15]).long(),\
                                    8:torch.Tensor([16]).long(),\
                                    9:torch.Tensor([17,18]).long(),\
                                    10:torch.Tensor([19]).long(),\
                                }
        elif 't2' in train_set:
            
            label_mapping={0:0,1:0,      21:0,22:0,\
                                2:1,3:1,      23:1,\
                                4:2,5:2,6:2,  24:2,25:2, \
                                7:3,           26:3,\
                                8:4  ,9:4,     27:4,28:4,      \
                                10:5, 11:5,       29:5   ,      \
                                12:6 , 13:6,      30:6,31:6,32:6,      \
                                14:7, 15:7,       33:7,34:7  ,   \
                                16:8,           37:8,\
                                17:9,18:9,         35:9,36:9,
                                19:10,                     38:10,39:10,\
                                20:11,
                                }
            
            
            
            family_mapping={0:torch.Tensor([0,1,  21,22  ]).long(),\
                                1:torch.Tensor([2,3,   23]).long(),  \
                                2:torch.Tensor([4,5,6,    24,25]).long(),\
                                    3:torch.Tensor([7,   26]).long(),\
                                    4:torch.Tensor([8,9,   27,28]).long(),\
                                    5:torch.Tensor([10,11,   29]).long(),\
                                    6:torch.Tensor([12,13,   30,31,32]).long(),\
                                    7:torch.Tensor([14,15,   33,34]).long(),\
                                    8:torch.Tensor([16,       37]).long(),\
                                    9:torch.Tensor([17,18,    35,36]).long(),\
                                    10:torch.Tensor([19,   38,39]).long(),\
                                    
                                    11:torch.Tensor([20]).long(),
                                
                                }
        elif 't3' in train_set:
            
            label_mapping={0:0,1:0,            21:0,22:0,        40:0, 41:0,\
                                2:1,3:1,            23:1,             42:1,\
                                4:2,5:2,6:2,        24:2,25:2,        43:2,44:2,45:2, \
                                7:3,                 26:3,             46:3,           \
                                8:4  ,9:4,           27:4,28:4,        47:4,48:4,49:4, \
                                10:5, 11:5,          29:5   ,          50:5,51:5,    \
                                12:6 , 13:6,        30:6,31:6,32:6,   52:6,53:6,   \
                                14:7, 15:7,       
                                33:7,34:7  ,      54:7,              \
                                16:8,               37:8,             55:8,56:8,           \
                                17:9,18:9,          35:9,36:9,        57:9,      \
                                19:10,              38:10,39:10,       58:10,59:10,   \
                                20:11,
                                }
            
            
            family_mapping={0:torch.Tensor([0,1,      21,22 ,      40,41 ]).long(),\
                                1:torch.Tensor([2,3,       23,          42]).long(),  \
                                2:torch.Tensor([4,5,6,     24,25,       43,44,45]).long(),\
                                    3:torch.Tensor([7,       26,          46]).long(),\
                                    4:torch.Tensor([8,9,     27,28,       47,48,49]).long(),\
                                    5:torch.Tensor([10,11,   29,          50,51]).long(),\
                                    6:torch.Tensor([12,13,   30,31,32,    52,53]).long(),\
                                    7:torch.Tensor([14,15,   33,34,       54]).long(),\
                                    8:torch.Tensor([16,      37,          55,56]).long(),\
                                    9:torch.Tensor([17,18,  35,36,       57]).long(),\
                                    10:torch.Tensor([19,     38,39,        58,59]).long(),\
                                    
                                    11:torch.Tensor([20]).long(),
                                
                                }
        
        elif 't4' in train_set:
            label_mapping={0:0,1:0,            21:0,22:0,        40:0, 41:0,        60:0,61:0,\
                                2:1,3:1,            23:1,             42:1,              62:1,\
                                4:2,5:2,6:2,        24:2,25:2,        43:2,44:2,45:2,    63:2,64:2,   \
                                7:3,                 26:3,             46:3,              65:3,66:3,\
                                8:4  ,9:4,           27:4,28:4,        47:4,48:4,49:4,    67:4,68:4,69:4,      \
                                10:5, 11:5,          29:5   ,          50:5,51:5,         70:5,71:5,\
                                12:6 , 13:6,        30:6,31:6,32:6,   52:6,53:6,         72:6,73:6,74:6,\
                                14:7, 15:7,         33:7,34:7  ,      54:7,              75:7,  \
                                16:8,               37:8,             55:8,56:8,         76:8,  \
                                17:9,18:9,          35:9,36:9,        57:9,              77:9,\
                                19:10,              38:10,39:10,       58:10,59:10,      78:10,79:10,\
                                20:11,
                                }
            
            family_mapping={0:torch.Tensor([0,1,      21,22 ,      40,41,          60,61 ]).long(),\
                                1:torch.Tensor([2,3,       23,          42,             62]).long(),  \
                                2:torch.Tensor([4,5,6,     24,25,       43,44,45,       63,64]).long(),\
                                    3:torch.Tensor([7,       26,          46,             65,66,]).long(),\
                                    4:torch.Tensor([8,9,     27,28,       47,48,49,       67,68,69]).long(),\
                                    5:torch.Tensor([10,11,   29,          50,51,          70,71]).long(),\
                                    6:torch.Tensor([12,13,   30,31,32,    52,53,          72,73,74]).long(),\
                                    7:torch.Tensor([14,15,   33,34,       54,             75]).long(),\
                                    8:torch.Tensor([16,      37,          55,56,          76]).long(),\
                                    9:torch.Tensor([17,18,  35,36,       57,             77]).long(),\
                                    10:torch.Tensor([19,     38,39,        58,59,         78,79]).long(),\
                                    
                                    11:torch.Tensor([20]).long(),
                                
                                }

            
    elif 'TOWOD' in dataset:
        if 't1' in train_set:
            label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,\
                            2:1,7:1,9:1, 11:1,12:1, 16:1,\
                            4:2,\
                            8:3,10:3,15:3,17:3,\
                            14:4,\
                            19:5}
                            
                
                
                            
            family_mapping={0:torch.Tensor([0,1,3,5,6,13,18]).long(),\
                            1:torch.Tensor([2,7,9,11,12,16]).long(),  \
                            2:torch.Tensor([4]).long(),\
                                3:torch.Tensor([8,10,15,17]).long(),\
                                4:torch.Tensor([14]).long(),\
                                5:torch.Tensor([19]).long(),
                            }
            
        elif 't2' in train_set:
            label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,    20:0,\
                            2:1,7:1,9:1, 11:1,12:1, 16:1,                   26:1,27:1,28:1,29:1,\
                            4:2,\
                            8:3,10:3,15:3,17:3,\
                            14:4,\
                            19:5,\
                                21:6,22:6,23:6,24:6,25:6,
                                30:7,31:7,32:7,33:7,34:7,
                                35:8,36:8,37:8,38:8,39:8,
                                }


            family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                            1:torch.Tensor([2,7,9,11,12,16,                  26,27,28,29 ]).long(),  \
                            2:torch.Tensor([4]).long(),\
                                3:torch.Tensor([8,10,15,17]).long(),\
                                4:torch.Tensor([14]).long(),\
                                5:torch.Tensor([19]).long(),
                                    6:torch.Tensor([                               21,22,23,24,25]).long(),
                                    7:torch.Tensor([                               30,31,32,33,34]).long(),
                                    8:torch.Tensor([                               35,36,37,38,39]).long(),
                            }
        elif 't3' in train_set:
                label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,    20:0,\
                            2:1,7:1,9:1, 11:1,12:1, 16:1,                   26:1,27:1,28:1,29:1,\
                            4:2,\
                            8:3,10:3,15:3,17:3,\
                            14:4,\
                            19:5,\
                                21:6,22:6,23:6,24:6,25:6,
                                30:7,31:7,32:7,33:7,34:7,
                                35:8,36:8,37:8,38:8,39:8,
                                    40:9,41:9,42:9,43:9,44:9,45:9,46:9,47:9,48:9,49:9, \
                                50:10,51:10,52:10,53:10,54:10,55:10,56:10,57:10,58:10,59:10,\
                                }
                
                
                
                family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                            1:torch.Tensor([2,7,9,11,12,16,                26,27,28,29 ]).long(),  \
                            2:torch.Tensor([4]).long(),\
                                3:torch.Tensor([8,10,15,17]).long(),\
                                4:torch.Tensor([14]).long(),\
                                5:torch.Tensor([19]).long(),
                                    6:torch.Tensor([                               21,22,23,24,25]).long(),
                                    7:torch.Tensor([                               30,31,32,33,34]).long(),
                                    8:torch.Tensor([                               35,36,37,38,39]).long(),
                                    9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                    10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                            }
        elif 't4' in train_set:
                label_mapping={0:0,1:0, 3:0, 5:0 , 6:0 , 13:0, 18:0,    20:0,\
                            2:1,7:1,9:1, 11:1,12:1, 16:1,                   26:1,27:1,28:1,29:1,\
                            4:2,                                                                          74:2,75:2,76:2,77:2,78:2,79:2,\
                            8:3,10:3,15:3,17:3,                                                           60:3,61:3, \
                            14:4,\
                            19:5,                                                                          62:5,63:5,64:5,65:5,66:5,\
                                21:6,22:6,23:6,24:6,25:6,
                                30:7,31:7,32:7,33:7,34:7,
                                35:8,36:8,37:8,38:8,39:8,
                                    40:9,41:9,42:9,43:9,44:9,45:9,46:9,47:9,48:9,49:9, \
                                50:10,51:10,52:10,53:10,54:10,55:10,56:10,57:10,58:10,59:10,\
                                    67:11,68:11,69:11,70:11,71:11,72:11,73:11,
                                }
            
            
            
                family_mapping={0:torch.Tensor([0,1,3,5,6,13,18,              20]).long(),\
                            1:torch.Tensor([2,7,9,11,12,16,               26,27,28,29 ]).long(),  \
                            2:torch.Tensor([4,                                                                                        74,75,76,77,78,79]).long(),\
                                3:torch.Tensor([8,10,15,17,                                                                             60,61     ]).long(),\
                                4:torch.Tensor([14]).long(),\
                                5:torch.Tensor([19,                                                                                     62,63,64,65,66]).long(),
                                    6:torch.Tensor([                               21,22,23,24,25]).long(),
                                    7:torch.Tensor([                               30,31,32,33,34]).long(),
                                    8:torch.Tensor([                               35,36,37,38,39]).long(),
                                    9:torch.Tensor([                                40,41,42,43,44,45,46,47,48,49]).long(),\
                                    10:torch.Tensor([                                50,51,52,53,54,55,56,57,58,59]).long(),
                                    11:torch.Tensor([                                                                                    67,68,69,70,71,72,73]).long(),
                            }
    
    
    elif 'OWDETR' in dataset:
            if 't1' in train_set:
                label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                    2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                    18:2,\
                                }
                
                family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\
                                }
            elif 't2' in train_set:
                
                label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                    2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                    18:2,\
                                        19:3,20:3,21:3,22:3,23:3,\
                                    24:4,25:4,26:4,37:4,38:4,39:4,\
                                    27:5,28:5,29:5,30:5,31:5,\
                                    32:6,33:6,34:6,35:6,36:6,\
                                }
                
                
                
                family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\

                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),

                                }
            elif 't3' in train_set:
                label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                    2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                    18:2,\
                                        19:3,20:3,21:3,22:3,23:3,\
                                    24:4,25:4,26:4,37:4,38:4,39:4,\
                                    27:5,28:5,29:5,30:5,31:5,\
                                    32:6,33:6,34:6,35:6,36:6,\
                                    40:7,41:7,42:7,43:7,44:7,45:7,46:7,47:7,48:7,49:7,\
                                    50:8,51:8,52:8,53:8,54:8,55:8,56:8,57:8,58:8,59:8,\
                                }
                
                family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\

                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),

                                }
            elif 't4' in train_set:
                
                label_mapping={0:0,1:0,3:0,4:0,5:0,10:0,12:0,17:0, \
                                    2:1,6:1,7:1,8:1,9:1,11:1,13:1,14:1,15:1,16:1,\
                                    18:2,\
                                        19:3,20:3,21:3,22:3,23:3,\
                                    24:4,25:4,26:4,37:4,38:4,39:4,\
                                    27:5,28:5,29:5,30:5,31:5,\
                                    32:6,33:6,34:6,35:6,36:6,\
                                    40:7,41:7,42:7,43:7,44:7,45:7,46:7,47:7,48:7,49:7,\
                                    50:8,51:8,52:8,53:8,54:8,55:8,56:8,57:8,58:8,59:8,\
                                    
                                    60:9,61:9,62:9,63:9,64:9,78:9,\
                                    65:10,66:10,67:10,68:10,69:10,70:10,71:10,\
                                    72:11,73:11,74:11,75:11,76:11,77:11,79:11
                                }
                
                family_mapping={0:torch.Tensor([0,1,3,4,5,10,12,17]).long(),\
                                1:torch.Tensor([2,6,7,8,9,11,13,14,15,16]).long(),  \
                                2:torch.Tensor([18]).long(),\

                                3:torch.Tensor([                                 19,20,21,22,23]).long(),\
                                4:torch.Tensor([                                 24,25,26,37,38,39]).long(),
                                5:torch.Tensor([                                 27,28,29,30,31]).long(),
                                6:torch.Tensor([                                 32,33,34,35,36]).long(),
                                7:torch.Tensor([                                                        40,41,42,43,44,45,46,47,48,49]).long(),
                                8:torch.Tensor([                                                        50,51,52,53,54,55,56,57,58,59]).long(),

                                9:torch.Tensor([                                                                                             60,61,62,63,64,78]).long(),
                                10:torch.Tensor([                                                                                            65,66,67,68,69,70,71]).long(),
                                11:torch.Tensor([                                                                                            72,73,74,75,76,77,79]).long(),

                                }

    return label_mapping, family_mapping

def convert_class_name_to_superclass(labels, class_mapping, family_mapping, hyp_dataset_class_names):
    """
    VOC_COCO_CLASS_NAMES: dictionary of shape {dataset_name: [class_name1, class_name2, ...]}
    labels: numpy array of shape (N,)
    class_mapping: dictionary of shape {class_index: superclass_index}
    family_mapping: dictionary of shape {superclass_index: [class_index1, class_index2, ...]}
    return: numpy array of shape (N,)
    """    
    
    label_classes = set(labels.tolist())
    defined_superclass_names = {}
    for label_class in label_classes:
        assert label_class in hyp_dataset_class_names
        label_superclass = class_mapping[hyp_dataset_class_names.index(label_class)]
        if label_superclass not in defined_superclass_names:
            defined_superclass_names[label_superclass] = []
        defined_superclass_names[label_superclass].append(label_class)
    for key in defined_superclass_names.keys():
        defined_superclass_names[key] = '_'.join(defined_superclass_names[key])

    transform_labels = []
    for label in labels:
        label_superclass = class_mapping[hyp_dataset_class_names.index(label)]
        transform_labels.append(defined_superclass_names[label_superclass])
        
    # for key in family_mapping.keys():
    #     for ind in family_mapping[key]:
    #         print(key, ind, hyp_dataset_class_names[int(ind)])
    
    return np.array(transform_labels)


def load_obj_features(objfeatures_filename, background_features_filename=None):
    with h5py.File(f'/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/{objfeatures_filename}.h5', 'r') as file:
        with h5py.File(f'/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/{objfeatures_filename}_class_name.h5', 'r') as class_name_file:
            print(len(file.keys()))
            layers_obj_features = {}
            layers_obj_features_class_name = []
            
            # Load object data
            for idx, sample_key in enumerate(file.keys()):
                class_names = class_name_file[sample_key][:]
                class_names = [name.decode('utf-8') for name in class_names]
                layers_obj_features_class_name.extend(class_names)
                
                for key in file[sample_key].keys():
                    for subkey in file[sample_key][key].keys():
                        if subkey not in layers_obj_features:
                            layers_obj_features[subkey] = []
                        layers_obj_features[subkey].append(np.array(file[sample_key][key][subkey]))

            # Load background features if provided
            if background_features_filename is not None:
                with h5py.File(f'/home/khoadv/projects/OOD_OD/PROB_Exploring/data/OWOD/ObjFeatures/{background_features_filename}.h5', 'r') as bg_file:
                    print(f"Loading background features from {background_features_filename}")
                    for idx, sample_key in enumerate(bg_file.keys()):
                        # Add "background" label for all background features
                        
                        add_label_idx = False
                        
                        for key in bg_file[sample_key].keys():
                            for subkey in bg_file[sample_key][key].keys():
                                assert subkey in layers_obj_features
                                num_bg_samples = bg_file[sample_key][key][subkey].shape[0]
                                if not add_label_idx:
                                    layers_obj_features_class_name.extend(['background'] * num_bg_samples)
                                    add_label_idx = True
                                layers_obj_features[subkey].append(np.array(bg_file[sample_key][key][subkey]))

            # Post processing            
            for subkey in layers_obj_features.keys():
                layers_obj_features[subkey] = np.concatenate(layers_obj_features[subkey], axis=0)
                print('layers_obj_features[subkey]', layers_obj_features[subkey].shape)
            layers_obj_features_class_name = np.array(layers_obj_features_class_name)
            print('layers_obj_features_class_name', layers_obj_features_class_name.shape)

    return layers_obj_features, layers_obj_features_class_name


def random_sampling(data, labels, m):
    np.random.seed(42)
    if labels.shape[0] < m:
        return data, labels
    indices = np.random.choice(data.shape[0], m, replace=False)
    sampled_data = data[indices]
    sampled_labels = labels[indices]
    return sampled_data, sampled_labels

def random_sampling_except_unknown(data, labels, m):
    np.random.seed(42)
    known_indices = np.where(labels != 'unknown')[0]
    if known_indices.shape[0] >= m:
        known_indices = np.random.choice(known_indices, m, replace=False)
    sampled_data = data[known_indices]
    sampled_labels = labels[known_indices]
    return sampled_data, sampled_labels

def random_sampling_with_background(data, labels, m):
    """Sample m points, keeping background separate but limiting its count"""
    np.random.seed(42)
    bg_indices = np.where(labels == 'background')[0]
    obj_indices = np.where(labels != 'background')[0]
    
    # Sample background (limit to m/4 or all if less)
    bg_sample_size = min(len(bg_indices), m // 4)
    if bg_sample_size > 0:
        bg_sampled = np.random.choice(bg_indices, bg_sample_size, replace=False)
    else:
        bg_sampled = np.array([], dtype=int)
    
    # Sample objects for the rest
    obj_sample_size = m - len(bg_sampled)
    if len(obj_indices) >= obj_sample_size:
        obj_sampled = np.random.choice(obj_indices, obj_sample_size, replace=False)
    else:
        obj_sampled = obj_indices
    
    # Combine
    all_indices = np.concatenate([bg_sampled, obj_sampled])
    return data[all_indices], labels[all_indices]


def tsne_visualization(layers_obj_features, layers_obj_features_class_name, random_sampling_function, save_path_lambda, super_class_transform=None):

        for subkey, obj_features in layers_obj_features.items():
            if '_in' in subkey: continue
            
            print(subkey, layers_obj_features[subkey].shape)

            sampled_data, sampled_labels = random_sampling_function(layers_obj_features[subkey], layers_obj_features_class_name, m=5000)
            
            # Don't apply super_class_transform to background
            if super_class_transform is not None:
                bg_mask = sampled_labels == 'background'
                unknown_mask = sampled_labels == 'unknown'
                obj_mask = ~(bg_mask | unknown_mask)
                transformed_labels = sampled_labels.copy()
                if obj_mask.sum() > 0:
                    transformed_labels[obj_mask] = convert_class_name_to_superclass(
                        sampled_labels[obj_mask], 
                        super_class_transform['class_mapping'], 
                        super_class_transform['family_mapping'], 
                        super_class_transform['hyp_dataset_class_names']
                    )
                sampled_labels = transformed_labels
            
            # Get unique classes and create a mapping to numeric labels
            unique_classes = np.unique(sampled_labels)
            class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
            print(f"Number of unique classes: {len(unique_classes)}")

            # Convert class names to numeric for coloring
            numeric_labels = np.array([class_to_idx[lbl] for lbl in sampled_labels])

            # Apply t-SNE to reduce to 2D
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            data_2d = tsne.fit_transform(sampled_data)

            # Plot with class-based colors
            plt.figure(figsize=(12, 10))
            
            # Create discrete colormap with exact number of colors needed
            cmap = plt.cm.get_cmap('tab20', len(unique_classes))
            
            scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], 
                                c=numeric_labels, 
                                cmap=cmap,
                                vmin=-0.5,
                                vmax=len(unique_classes) - 0.5,
                                alpha=0.6, 
                                s=10)
            
            # Add colorbar or legend with larger text
            cbar = plt.colorbar(scatter, ticks=range(len(unique_classes)))
            cbar.ax.set_yticklabels(unique_classes, fontsize=14)
            cbar.set_label('Class', fontsize=16)
            
            plt.xlabel('t-SNE 1', fontsize=16)
            plt.ylabel('t-SNE 2', fontsize=16)
            plt.title(f't-SNE Visualization - {subkey}', fontsize=18)
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            plt.tight_layout()
            plt.savefig(save_path_lambda(subkey), dpi=150)
            plt.close()


if __name__ == '__main__':
    objfeatures_filename = 'objfeatures_V10_IoU06'
    background_features_filename = 'objfeatures_V10_less_IoU01'
    
    # 1.0 TSNE Visualization with background
    layers_obj_features, layers_obj_features_class_name = load_obj_features(objfeatures_filename, background_features_filename)
    tsne_visualization(layers_obj_features, layers_obj_features_class_name, random_sampling_with_background, lambda x: f'../trash/tsne_plot_IoU06_{x}.png')

    # 1.1 TSNE Visualization for super-class with background
    class_mapping, family_mapping = Hyp_OW_Superclass_mapping('TOWOD', 't4')
    hyp_dataset_class_names = Hyp_OW_class_name()['TOWOD']
    super_class_transform = {'class_mapping': class_mapping, 'family_mapping': family_mapping, 'hyp_dataset_class_names': hyp_dataset_class_names}
    layers_obj_features, layers_obj_features_class_name = load_obj_features(objfeatures_filename, background_features_filename)
    tsne_visualization(layers_obj_features, layers_obj_features_class_name, random_sampling_with_background, lambda x: f'../trash/tsne_plot_IoU06_{x}_super_class.png', super_class_transform)

    # # 1.2 Hyperbolic visualization
    # layers_obj_features, layers_obj_features_class_name = load_obj_features(objfeatures_filename, background_features_filename)
    # hyperbolic_features = to_hyperbolic(layers_obj_features)
    # tsne_visualization(hyperbolic_features, layers_obj_features_class_name, random_sampling_with_background, lambda x: f'../trash/tsne_plot_IoU06_{x}_hyperbolic.png')
    
    # # 1.3 Hyperbolic visualization for super-class
    # class_mapping, family_mapping = Hyp_OW_Superclass_mapping('TOWOD', 't4')
    # hyp_dataset_class_names = Hyp_OW_class_name()['TOWOD']
    # super_class_transform = {'class_mapping': class_mapping, 'family_mapping': family_mapping, 'hyp_dataset_class_names': hyp_dataset_class_names}
    # layers_obj_features, layers_obj_features_class_name = load_obj_features(objfeatures_filename, background_features_filename)
    # hyperbolic_features = to_hyperbolic(layers_obj_features)
    # tsne_visualization(hyperbolic_features, layers_obj_features_class_name, random_sampling_with_background, lambda x: f'../trash/tsne_plot_IoU06_{x}_hyperbolic_super_class.png', super_class_transform)
    
    pass