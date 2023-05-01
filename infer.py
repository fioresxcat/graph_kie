import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import argparse
import json
import pdb
import shutil
from model import RGCN_Model
from dataset import GraphDataModule, GraphDataset
import yaml
from pathlib import Path
from my_utils import *
from label_list import all_label_list
from sklearn.metrics import classification_report
from bpemb import BPEmb


def load_model(general_cfg, model_cfg, n_classes, ckpt_path=None):
    model_type = general_cfg['options']['model_type']
    if model_type == 'rgcn':
        if ckpt_path is not None:
            model = RGCN_Model.load_from_checkpoint(
                        checkpoint_path=ckpt_path,
                        general_config=general_cfg, 
                        model_config=model_cfg, 
                        n_classes=n_classes
                    )
        else:
            model = RGCN_Model(general_config=general_cfg, model_config=model_cfg, n_classes=n_classes)
    else:
        raise ValueError(f'Model type {model_type} is not supported yet')    
    
    return model


def get_input_from_json(json_fp, word_encoder, use_emb, emb_range):
    with open(json_fp, 'r') as f:
            json_data = json.load(f)
    img_h, img_w = json_data['imageHeight'], json_data['imageWidth']

    # get node features
    x_indexes = [] # list of all x_indexes of all nodes in graph (each node has an x_index)
    y_indexes = [] # list of all y_indexes of all nodes in graph (each node has an y_index)
    text_features = [] # list of all features of all nodes in graph (each node has a feature)

    bb2label, bb2text, rbbs, bbs2idx_sorted, sorted_indices = sort_json(json_data)

    edges = []
    for row_idx, rbb in enumerate(rbbs):
        for bb_idx_in_row, bb in enumerate(rbb):  # duyet qua tung bb (tung node)
            # ----------------- process text feature -----------------
            text = bb2text[bb]
            if word_encoder.lang != 'vi':
                text = unidecode.unidecode(text)  # nếu hóa đơn ko dấu thì bpemb để tiếng việt hay tiếng anh ?
            bb_text_feature = get_manual_text_feature(text) + list(np.sum(word_encoder.embed(text), axis=0))
            text_features.append(bb_text_feature)

            # ----------------- process geometry feature -----------------
            xmin, ymin, xmax, ymax = get_bb_from_poly(bb)
            # augment box coord
            if use_emb: 
                # rescale coord at width=emb_range
                x_index = [int(xmin * emb_range / img_w), int(xmax * emb_range / img_w), int((xmax - xmin) * emb_range / img_w)]
                y_index = [int(ymin * emb_range / img_h), int(ymax * emb_range / img_h), int((ymax - ymin) * emb_range / img_h)]
            else:
                # normalize in rnage(0, 1)
                x_index = [float(xmin * 1.0 / img_w), float(xmax * 1.0 / img_w), float((xmax - xmin) * 1.0 / img_w)]
                y_index = [float(ymin * 1.0 / img_h), float(ymax * 1.0 / img_h), float((ymax - ymin) * 1.0 / img_h)]
            x_indexes.append(x_index)
            y_indexes.append(y_index)
            
            # ------------------------ build graph ----------------------
            # find right node
            right_node = rbb[bb_idx_in_row+1] if bb_idx_in_row < len(rbb) - 1 else None
            if right_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[right_node], 1])
                edges.append([bbs2idx_sorted[right_node], bbs2idx_sorted[bb], 2])
            
            # find left node
            left_node = rbb[bb_idx_in_row-1] if bb_idx_in_row > 0 else None
            if left_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[left_node], 2])
                edges.append([bbs2idx_sorted[left_node], bbs2idx_sorted[bb], 1])
            
            # find above node
            max_x_overlap = -1e9
            above_node = None
            if row_idx > 0:
                for prev_bb in rbbs[row_idx-1]:
                    xmax_prev_bb = max(prev_bb[2], prev_bb[4])
                    xmin_prev_bb = min(prev_bb[0], prev_bb[6])
                    x_overlap = (xmax_prev_bb - xmin_prev_bb) + (xmax-xmin) - (max(xmax_prev_bb, xmax) - min(xmin_prev_bb, xmin))
                    if x_overlap > max_x_overlap:
                        max_x_overlap = x_overlap
                        above_node = prev_bb
            if above_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[above_node], 4])
                edges.append([bbs2idx_sorted[above_node], bbs2idx_sorted[bb], 3])
            
            # find below node
            max_x_overlap = -1e9
            below_node = None
            if row_idx < len(rbbs) - 1:
                for next_bb in rbbs[row_idx+1]:
                    xmax_next_bb = max(next_bb[2], next_bb[4])
                    xmin_next_bb = min(next_bb[0], next_bb[2])
                    x_overlap = (xmax_next_bb - xmin_next_bb) + (xmax-xmin) - (max(xmax_next_bb, xmax) - min(xmin_next_bb, xmin))
                    if x_overlap > max_x_overlap:
                        max_x_overlap = x_overlap
                        below_node = next_bb
            if below_node:
                edges.append([bbs2idx_sorted[bb], bbs2idx_sorted[below_node], 3])
                edges.append([bbs2idx_sorted[below_node], bbs2idx_sorted[bb], 4])

    # 1 - right, 2 - left, 3 - down, 4  - up
    edges = torch.tensor(edges, dtype=torch.int32)
    edges = torch.unique(edges, dim=0, return_inverse=False)   # remove duplicate rows
    edge_index, edge_type = edges[:, :2], edges[:, -1]

    return torch.tensor(x_indexes, dtype=torch.int if use_emb else torch.float),  \
            torch.tensor(y_indexes, dtype=torch.int if use_emb else torch.float), \
            torch.tensor(text_features, dtype=torch.float), \
            edge_index.t().to(torch.int64), \
            edge_type, \



def inference(ckpt_path, src_dir=None, out_dir=None):
    os.makedirs(out_dir, exist_ok=True)
    ckpt_dir = Path(ckpt_path).parent
    with open(os.path.join(ckpt_dir, 'train_cfg.yaml')) as f:
        general_cfg = yaml.load(f, Loader=yaml.FullLoader)
    with open(os.path.join(ckpt_dir, 'model_cfg.yaml')) as f:
        model_cfg = yaml.load(f, Loader=yaml.FullLoader)

    label_list = all_label_list[general_cfg['data']['label_list']]
    word_encoder = BPEmb(**general_cfg['options']['word_encoder'])
    use_emb = general_cfg['options']['use_emb'],
    emb_range = general_cfg['options']['emb_range']
    model = load_model(general_cfg, model_cfg, n_classes=len(label_list), ckpt_path=ckpt_path)
    model.eval()

    json_files = sorted(list(Path(src_dir).rglob('*.json')))
    err_dict = {}
    all_trues, all_preds = [], []
    for i, json_fp in enumerate(json_files):
        json_data = json.load(open(json_fp))
        _, _, _, _, sorted_indices = sort_json(json_data)

        # model infer
        x_indexes, y_indexes, text_features, edge_index, edge_type = get_input_from_json(
            json_fp,
            word_encoder,
            use_emb,
            emb_range
        )
        out = model(x_indexes, y_indexes, text_features, edge_index, edge_type)
        preds = torch.argmax(out, dim=-1)

        # modify json_label
        for i, shape in enumerate(json_data['shapes']):
            pred_label = label_list[preds[sorted_indices[i]]]
            json_data['shapes'][i]['label'] = 'text'
            json_data['shapes'][i]['label'] = pred_label
            if pred_label != shape['label']:
                err_info = {
                    'box': [int(coord) for pt in shape['points'] for coord in pt],
                    'text': shape['text'],
                    'true_label': shape['label'],
                    'pred_label': pred_label
                }
                if str(json_fp) not in err_dict:
                    err_dict[str(json_fp)] = [err_info]
                else:
                    err_dict[str(json_fp)].append(err_info) 

            all_trues.append(label_list.index(shape['label']))
            all_preds.append(label_list.index(pred_label))

        # save
        with open(os.path.join(out_dir, json_fp.name), 'w') as f:
            json.dump(json_data, f)
        shutil.copy(json_fp.with_suffix('.jpg'), out_dir)
        print(f'Done {json_fp.name}')
    
    target_names = [label_list[i] for i in sorted(list(set(all_trues)))]
    report = classification_report(y_true=all_trues, y_pred=all_preds, target_names=target_names)
    print(report)
    with open(os.path.join(out_dir, 'error_dict.json'), 'w') as f:
        json.dump(err_dict, f)
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some arguments...')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to lightning checkpoint')
    parser.add_argument('--src_dir', type=str, required=True, help='path to inference dir')
    parser.add_argument('--out_dir', type=str, required=True, help='path to write result')

    args = parser.parse_args()
    inference(
        args.ckpt_path,
        args.src_dir,
        args.out_dir
    )