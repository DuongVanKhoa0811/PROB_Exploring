import torch.nn as nn
import torch
import numpy as np
from torchvision.ops import roi_align
from functools import partial
from itertools import combinations
from datasets.torchvision_datasets.open_world import VOC_COCO_CLASS_NAMES


threshold = 0.5
hook_version = 'v0_obj' # [v0, v1, v0_obj]


def combined_cnn_layer(hook_names, hook_index):
    combined_hook_names = {}
    number_to_string = {1: "one", 2: "two", 3: "three", 4: "four"}
    
    n_cnn_layers = hook_index['e_cnn_hook_idx'] - hook_index['s_cnn_hook_idx'] + 1
    for idx in range(1, n_cnn_layers+1):
        cnn_combinations_hook_names = list(combinations(hook_names[hook_index['s_cnn_hook_idx'] : hook_index['e_cnn_hook_idx']+1], idx))
        combined_hook_names[f'combined_{number_to_string[idx]}_cnn_layer_hook_names'] = []
        for cnn_combinations_hook_name in cnn_combinations_hook_names:
            if 's_tra_enc_hook_idx' in hook_index:
                for i_enc in range(hook_index['s_tra_enc_hook_idx'], hook_index['e_tra_enc_hook_idx']+1):
                    combined_hook_names[f'combined_{number_to_string[idx]}_cnn_layer_hook_names'].append(list(cnn_combinations_hook_name) + [hook_names[i_enc]])
            if 's_tra_dec_hook_idx' in hook_index:
                for i_dec in range(hook_index['s_tra_dec_hook_idx'], hook_index['e_tra_dec_hook_idx']+1):
                    combined_hook_names[f'combined_{number_to_string[idx]}_cnn_layer_hook_names'].append(list(cnn_combinations_hook_name) + [hook_names[i_dec]])
        
    return combined_hook_names

def collect_in_out_hook_names(block_layers):
    block_layers_with_in_out = []
    for block_layer in block_layers:
        block_layers_with_in_out += [block_layer + '_in', block_layer + '_out'] 
    
    return block_layers_with_in_out


### v0 Penultimate layer
hook_names_v0 = []
hook_names_v0.extend(collect_in_out_hook_names(['transformer.decoder.layers.5.norm3']))
hook_index_v0 = {
    's_tra_dec_hook_idx' : hook_names_v0.index('transformer.decoder.layers.5.norm3_in'),
    'e_tra_dec_hook_idx' : hook_names_v0.index('transformer.decoder.layers.5.norm3_out'),
}


### v0_obj Penultimate layer of the MLP projection
hook_names_v0_obj = []
hook_names_v0_obj.extend(collect_in_out_hook_names(['transformer.decoder.layers.5.norm3']))
hook_names_v0_obj.extend(collect_in_out_hook_names(['feature_projector.0.norm']))
hook_index_v0_obj = {
    's_tra_dec_hook_idx' : hook_names_v0_obj.index('transformer.decoder.layers.5.norm3_in'),
    'e_tra_dec_hook_idx' : hook_names_v0_obj.index('feature_projector.0.norm_out'),
}

### v1 Encoder + Decoder + SAFE
hook_names_v1 = []

## Encoder
for block_idx in range(6):
    
    block_layers = [
        f'transformer.encoder.layers.{block_idx}.self_attn.sampling_offsets',
        f'transformer.encoder.layers.{block_idx}.self_attn.attention_weights',
        f'transformer.encoder.layers.{block_idx}.self_attn.value_proj',
        f'transformer.encoder.layers.{block_idx}.self_attn.output_proj',
        f'transformer.encoder.layers.{block_idx}.norm1',
        f'transformer.encoder.layers.{block_idx}.linear1',
        f'transformer.encoder.layers.{block_idx}.linear2',
        f'transformer.encoder.layers.{block_idx}.norm2',
    ]
    block_layers_with_in_out = collect_in_out_hook_names(block_layers)
    hook_names_v1.extend(block_layers_with_in_out)

## Decoder
for block_idx in range(6):
    # Ignore dropout in the decoder
    block_layers = [
        f'transformer.decoder.layers.{block_idx}.cross_attn.sampling_offsets',
        f'transformer.decoder.layers.{block_idx}.cross_attn.attention_weights',
        f'transformer.decoder.layers.{block_idx}.cross_attn.value_proj',
        f'transformer.decoder.layers.{block_idx}.cross_attn.output_proj',
        # f'transformer.decoder.layers.{block_idx}.dropout1',
        f'transformer.decoder.layers.{block_idx}.norm1',
        # f'transformer.decoder.layers.{block_idx}.self_attn.out_proj',
        # f'transformer.decoder.layers.{block_idx}.dropout2',
        f'transformer.decoder.layers.{block_idx}.norm2',
        f'transformer.decoder.layers.{block_idx}.linear1',
        # f'transformer.decoder.layers.{block_idx}.dropout3',
        f'transformer.decoder.layers.{block_idx}.linear2',
        # f'transformer.decoder.layers.{block_idx}.dropout4',
        f'transformer.decoder.layers.{block_idx}.norm3',
        f'transformer.decoder.layers.{block_idx}.linear3',
        # f'transformer.decoder.layers.{block_idx}.dropout5',
        f'transformer.decoder.layers.{block_idx}.linear4',
        # f'transformer.decoder.layers.{block_idx}.dropout6',
        f'transformer.decoder.layers.{block_idx}.norm4',
    ]
    block_layers_with_in_out = collect_in_out_hook_names(block_layers)
    hook_names_v1.extend(block_layers_with_in_out)

## SAFE
hook_names_v1.extend(collect_in_out_hook_names(['backbone.0.body.layer1.0.downsample']))
hook_names_v1.extend(collect_in_out_hook_names(['backbone.0.body.layer2.0.downsample']))
hook_names_v1.extend(collect_in_out_hook_names(['backbone.0.body.layer3.0.downsample']))
hook_names_v1.extend(collect_in_out_hook_names(['backbone.0.body.layer4.0.downsample']))

hook_index_v1 = {
    's_cnn_hook_idx' : hook_names_v1.index('backbone.0.body.layer1.0.downsample_in'),
    'e_cnn_hook_idx' : hook_names_v1.index('backbone.0.body.layer4.0.downsample_out'),
    's_tra_enc_hook_idx' : hook_names_v1.index('transformer.encoder.layers.0.self_attn.sampling_offsets_in'),
    'e_tra_enc_hook_idx' : hook_names_v1.index('transformer.encoder.layers.5.norm2_out'),
    's_tra_dec_hook_idx' : hook_names_v1.index('transformer.decoder.layers.0.cross_attn.sampling_offsets_in'),
    'e_tra_dec_hook_idx' : hook_names_v1.index('transformer.decoder.layers.5.norm4_out'),
}


if hook_version == 'v0':
    hook_names = hook_names_v0
    hook_index = hook_index_v0
if hook_version == 'v0_obj':
    hook_names = hook_names_v0_obj
    hook_index = hook_index_v0_obj
elif hook_version == 'v1':
    hook_names = hook_names_v1
    hook_index = hook_index_v1
    # assert store_layer_features_seperate
else:
    raise ValueError(f'Invalid hook version: {hook_version}')


class featureTracker():
    def __init__(self, model, variant='DDETR'):
        self.variant = variant
        if "RCNN" in self.variant:
            model = model.model
        self.hook_model(model=model)


    @torch.no_grad()
    def __hook(self, model_self, inputs, outputs, idx):
        self.features[idx] = outputs
  
    @torch.no_grad()
    def __backward_hook(self, module, grad_input, grad_output, idx):
        self.gradients[idx] = grad_output[0]


    @torch.no_grad()
    def __hook_DDETR(self, model_self, inputs, outputs, idx, collect_input):
        if collect_input:
            assert len(inputs) == 1
            self.features[idx] = inputs[0]
        else: self.features[idx] = outputs
  
    @torch.no_grad()
    def __backward_hook_DDETR(self, module, grad_input, grad_output, idx, collect_input):
        if collect_input:
            self.gradients[idx] = grad_input[0]
        else: self.gradients[idx] = grad_output[0]
  

    @torch.no_grad()
    def hook_model(self, model):
        self.map_hook_names_to_idx = {}
        self.map_idx_to_hook_names = {}
        if self.variant == 'DDETR':
      
                for idx, (n, m) in enumerate(model.named_modules()):
                    print('tracker', idx, n)
      
                ### Specific task, penultimate layer features
                # hook_queue = []
                # for n, m in model.named_modules():
                #     if n == 'transformer.decoder.layers.5.norm3':
                #         hook_queue.append((m, False))
    
                hook_queue = []
                hook_count = 0
                for n, m in model.named_modules():
                    if n == hook_names[hook_count].replace('_in', '').replace('_out', ''): 
                        print(f'Prepare register hook for {hook_names[hook_count]}')
                        
                        if '_in' == hook_names[hook_count][-3:]:
                            hook_queue.append((m, True))
                            self.map_hook_names_to_idx[hook_names[hook_count]] = hook_count
                            hook_count += 1
                            if hook_count >= len(hook_names): break
    
                            if '_out' == hook_names[hook_count][-4:]:
                                print(f'Prepare register hook for {hook_names[hook_count]}')
                                hook_queue.append((m, False))
                                self.map_hook_names_to_idx[hook_names[hook_count]] = hook_count
                                hook_count += 1
                                if hook_count >= len(hook_names): break
                        else:
                            assert False, 'Temporary error'
    
                print('self.map_hook_names_to_idx', self.map_hook_names_to_idx)
                self.map_idx_to_hook_names = {v: k for k, v in self.map_hook_names_to_idx.items()}
    
    
        elif self.variant == 'DETR':
            hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Sequential) and 'downsample' in n]
        elif self.variant == 'RCNN-RGX4':
            hook_queue = [model.backbone.bottom_up.s1.b1.bn, model.backbone.bottom_up.s2.b1.bn, model.backbone.bottom_up.s3.b1.bn, model.backbone.bottom_up.s4.b1.bn]
        elif self.variant == 'RCNN-RN50':
            hook_queue = [m for n, m in model.named_modules() if isinstance(m, nn.Conv2d) and 'shortcut' in n]
        else: assert False
        
        self.features = [0] * len(hook_queue)
        self.gradients = [0] * len(hook_queue)
        self.out_size = []

        if self.variant == 'DDETR':
            for idx, (module, collect_input) in enumerate(hook_queue):
                hook_fn = partial(self.__hook_DDETR, idx=idx, collect_input=collect_input)
                backward_hook_fn = partial(self.__backward_hook_DDETR, idx=idx, collect_input=collect_input)
                module.register_forward_hook(hook_fn)
                module.register_backward_hook(backward_hook_fn)
            print('Complete register for modules!')
        else:
            for idx, module in enumerate(hook_queue):
                hook_fn = partial(self.__hook, idx=idx)
                backward_hook_fn = partial(self.__backward_hook, idx=idx)
                module.register_forward_hook(hook_fn)
                module.register_backward_hook(backward_hook_fn)
            print('Complete register for modules!')

        
    @torch.no_grad()
    def roi_features(self, rois, input_h):
        det_feats = []
        for feat in self.features:
            _, _, h, _ = feat.size()
            scale = h/input_h
            feat = roi_align(feat, rois, (1, 1), scale).mean(dim=(2, 3))
            det_feats.append(feat)
        return torch.cat(det_feats, dim=1)

    @torch.no_grad()
    def roi_backward_features(self, rois, input_h):
        grad_feats = []
        for grad in self.gradients:
            _, _, h, _ = grad.size()
            scale = h / input_h
            grad = roi_align(grad, rois, (1, 1), scale).mean(dim=(2, 3))
            grad_feats.append(grad)
        return torch.cat(grad_feats, dim=1)

    @torch.no_grad()
    def plus_features_gradients(self, eps):
        for i in range(len(self.features)):
            self.features[i] = self.features[i] + eps*self.gradients[i].sign()
   

    @torch.no_grad()
    def flush_features(self):
        self.features = [0] * len(self.features)
  
    @torch.no_grad()
    def flush_gradients(self):
        self.gradients = [0] * len(self.gradients)


def extract_obj(outputs, tracker, invalid_cls_logits, temperature, pred_per_im, dataset_name):
    
    examples_top_query_features = {'decoder_object_queries': {hook_names[i]: [] for i in range(hook_index['s_tra_dec_hook_idx'], hook_index['e_tra_dec_hook_idx'] + 1)}}
        

    ################ Object-specific features - object queries in the decoder ################
    if hook_version in ['v0', 'v1', 'v0_obj']:
        
        ### Collect topk result
        out_logits, pred_obj = outputs['pred_logits'], outputs['pred_obj']
        out_logits[:,:, invalid_cls_logits] = -10e10
        obj_prob = torch.exp(-temperature*pred_obj).unsqueeze(-1)
        prob = obj_prob*out_logits.sigmoid()
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), pred_per_im, dim=1)
        topk_query_index = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2]
        layers_topk_query_features = [] # LxBx100xC
        
        # Normal task
        for idx in range(hook_index['s_tra_dec_hook_idx'], hook_index['e_tra_dec_hook_idx'] + 1,1):
            layers_topk_query_features.append(torch.gather(tracker.features[idx], 1, topk_query_index.unsqueeze(-1).repeat(1, 1, tracker.features[idx].shape[-1])))

        ### Convert the topk result to final result based on the threshold
        scores = topk_values
        for batch_idx in range(outputs['pred_logits'].shape[0]):
            scores_example_mask = scores[batch_idx] > threshold
            
            assert batch_idx == 0
            no_objects = bool(scores_example_mask.sum() < 1)
            class_name = [VOC_COCO_CLASS_NAMES[dataset_name][int(label.item())] for label in labels[batch_idx][scores_example_mask]]
            
            example_top_query_features = [] # L x N_objects x C
            for i_layer_topk_query_features in layers_topk_query_features:
                example_top_query_features.append(i_layer_topk_query_features[batch_idx][scores_example_mask])
			# Normal task
            for i_layer in range(hook_index['s_tra_dec_hook_idx'], hook_index['e_tra_dec_hook_idx'] + 1):
                examples_top_query_features['decoder_object_queries'][hook_names[i_layer]].append(example_top_query_features[i_layer - hook_index['s_tra_dec_hook_idx']].to('cpu'))
				
    ########################################################################################
    tracker.flush_features()
    
    return examples_top_query_features, no_objects, class_name


def save_obj_features(features, dset_file, index):
    group = dset_file.create_group(f'{index}')
            
    for key, value in features.items():
        if isinstance(value, list): # decoder_object_queries, encoder_roi_align
            assert len(value) == 1, "Expected a single sample"
            group.create_dataset(f'{key}', data=np.array(value[0]))
        elif isinstance(value, dict):
            subgroup = group.create_group(f'{key}')
            for subkey, subvalue in value.items():
                assert len(subvalue) == 1, "Expected a single sample"
                subgroup.create_dataset(f'{subkey}', data=np.array(subvalue[0]))
