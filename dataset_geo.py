import os
import re
import json
import math
import numpy as np
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch.utils.data import Dataset, DataLoader
from bpemb import BPEmb
from my_utils import *
import unidecode
from pathlib import Path
from typing import Optional, Tuple, List
import torch
import pytorch_lightning as pl
from label_list import all_label_list
# from torch_geometric.data import Dataset, DataLoader


def get_bb_from_poly(poly: Tuple):
    x1, y1, x2, y2, x3, y3, x4, y4 = poly    # tl -> tr -> br -> bl
    xmin = min (x1, x4)
    xmax = max (x2, x3)
    ymin = min (y1, y2)
    ymax = max (y3, y4)

    return xmin, ymin, xmax, ymax


def augment_box(xmin, ymin, xmax, ymax, img_w, img_h, percent=5):
    box_w = xmax - xmin
    box_h = ymax - ymin
    xmin += np.random.randint(-int(box_w/100*percent), int(box_w/100*percent)+1)
    xmax += np.random.randint(-int(box_w/100*percent), int(box_w/100*percent)+1)
    ymin += np.random.randint(-int(box_h/100*percent), int(box_h/100*percent)+1)
    ymax += np.random.randint(-int(box_h/100*percent), int(box_h/100*percent)+1)

    xmin = min(img_w, max(0, xmin))
    ymin = min(img_h, max(0, ymin))
    xmax = min(img_w, max(0, xmax))
    ymax = min(img_h, max(0, ymax))

    return xmin, ymin, xmax, ymax



def get_list_json(data_dir: str, max_sample=int(1e4), shuffle=False):
    print('max sample: ', max_sample)
    ls_json_fp = sorted(list(Path(data_dir).rglob('*.json')))
    if shuffle:
        np.random.shuffle(ls_json_fp)

    return ls_json_fp[:max_sample]


def random_drop_shape(shapes: List, num_general_drop: int, num_field_drop: int, outlier_label='text'):
    ls_idx2drop = []
    if np.random.rand() < 0.5:  # chon random ca text va non-text
        ls_idx2drop = np.random.randint(0, len(shapes), size=num_general_drop)
    else:   # chi drop non-text
        non_text_indices = [i for i, shape in enumerate(shapes) if shape['label'] != outlier_label]
        if len(non_text_indices) > 5:
            ls_idx2drop = np.random.choice(non_text_indices, min(num_field_drop, len(non_text_indices)//3))
    shapes = [shape for i, shape in enumerate(shapes) if i not in ls_idx2drop]

    return shapes


def augment_pad(shapes, img_w, img_h, max_offset=15):
    """
        add offset width and height to image
    """
    offset_x1, offset_x2, offset_y1, offset_y2 = [np.random.randint(-max_offset, max_offset), np.random.randint(-max_offset, max_offset),  \
                                                  np.random.randint(-max_offset, max_offset), np.random.randint(-max_offset, max_offset)]
    img_w += offset_x1 + offset_x2
    img_h += offset_y1 + offset_y2

    for i, shape in enumerate(shapes):
        for pt_idx, pt in enumerate(shape['points']):
            pt[0] = min(img_w, max(0, pt[0]+offset_x1))
            pt[1] = min(img_h, max(0, pt[1]+offset_y1))
            shapes[i]['points'][pt_idx] = [pt[0], pt[1]]

    return shapes, img_w, img_h


def get_manual_text_feature(text: str):
    feature = []

    # có phải ngày tháng không
    feature.append(int(re.search('(\d{1,2})\/(\d{1,2})\/(\d{4})', text) != None))

    # co phai gio khong
    feature.append(int(re.search('(\d{1,2}):(\d{1,2})', text) != None))
        
    # có phải ma hang hoa khong
    feature.append(int(re.search('^\d+$', text) != None and len(text) > 5))

    # có phải tiền dương không
    feature.append(int(re.search('^\d{1,3}(\,\d{3})*(\,00)+$', text.replace('.', ',')) != None or re.search('^\d{1,3}(\,\d{3})+$', text.replace('.', ',')) != None))
    
    # co phai tien am khong
    feature.append(int(text.startswith('-') and re.search('^[\d(\,)]+$', text[1:].replace('.', ',')) != None and len(text) >= 3))

    # có phải uppercase
    feature.append(int(text.isupper()))

    # có phải title
    feature.append(int(text.istitle()))

    # có phải lowercase
    feature.append(int(text.islower()))
    
    # có phải chỉ chứa chữ in hoa và số
    feature.append(int(re.search('^[A-Z0-9]+$', text) != None))

    # chỉ có số
    feature.append(int(re.search('^\d+$', text) != None))

    # chỉ có chữ cái
    feature.append(int(re.search('^[a-zA-Z]+$', text) != None))

    # chi co chu hoac so
    feature.append(int(re.search('^[a-zA-Z0-9]+$', text) != None))

    # chỉ có số và dấu
    feature.append(int(re.search('^[\d|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) != None))

    # chỉ có chữ và dấu
    feature.append(int(re.search('^[a-zA-Z|\-|\'|,|\(|\)|.|\/|&|:|+|~|*|\||_|>|@|%]+$', text) != None))

    return feature


    


class GraphDataset(Dataset):
    def __init__(self, config, mode='train'):
        self.config = config
        self.data_dir = config['data']['train_dir'] if mode == 'train' else config['data']['val_dir']
        self.n_labels = len(all_label_list[config['data']['label_list']])
        self.label_list = all_label_list[config['data']['label_list']]
        self.json_files = get_list_json(self.data_dir, config['data']['max_sample'])
        self.mode = mode
        self.use_emb = config['options']['use_emb']
        self.bpemb = BPEmb(lang="en", dim=100, vs=5000)
        self.emb_range = config['model']['embedding_range']

        self.print_dataset_info()
        self.invalid_labels = self.check_label_valid()


    def print_dataset_info(self):
        print(f'Dataset info: {len(self.json_files)} json files, {self.n_labels} labels')
        print(f'Label list: {self.label_list}')
        
    
    def check_label_valid(self):
        print('------------------------ Checking labels ----------------------------')
        ls_json_fp = list(Path(self.data_dir).rglob('*.json'))
        invalid_labels = []
        for jp in ls_json_fp:
            json_data = json.load(open(jp))
            for shape in json_data['shapes']:
                if shape['label'] not in all_label_list[self.config['data']['label_list']]:
                    # print(f'{jp} has outlier label: ', shape['label'])
                    invalid_labels.append(shape['label'])
        if len(invalid_labels) == 0:
            print('All labels are valid !')

        return invalid_labels


    def len(self):
        return len(self.json_files)


    def get(self, index):
        json_fp = self.json_files[index]
        with open(json_fp, 'r') as f:
            json_data = json.load(f)
        img_h, img_w = json_data['imageHeight'], json_data['imageWidth']

        # get node features
        nodes = []  # list of all node in graph
        x_indexes = [] # list of all x_indexes of all nodes in graph (each node has an x_index)
        y_indexes = [] # list of all y_indexes of all nodes in graph (each node has an y_index)
        text_features = [] # list of all features of all nodes in graph (each node has a feature)

        if self.mode == 'train':
            # random drop out boxes
            if len(json_data['shapes']) > 5 and np.random.rand() < 0.3:
                json_data['shapes'] = random_drop_shape(
                    json_data['shapes'], 
                    num_general_drop=np.random.randint(0, len(json_data['shapes'])//10),
                    num_field_drop=np.random.randint(3, 10)
                )
            # augment pad
            if np.random.rand() < 0.3:
                json_data['shapes'], img_w, img_h = augment_pad(json_data['shapes'], img_w, img_h, max_offset=15)

        # duyệt qua từng bounding box trong ảnh (mỗi shape là 1 bb)
        bbs, labels, texts = [], [], []
        for shape in json_data['shapes']:
            x1, y1 = shape['points'][0]  # tl
            x2, y2 = shape['points'][1]  # tr
            x3, y3 = shape['points'][2]  # br
            x4, y4 = shape['points'][3]  # bl
            bbs.append(tuple(int(i) for i in (x1,y1,x2,y2,x3,y3,x4,y4)))
            labels.append(shape['label'])
            texts.append(shape['text'])

        bb2label = dict(zip(bbs, labels))   # theo thu tu truyen vao trong data['shapes']
        bb2text = dict(zip(bbs, texts))
        bb2idx_original = {tuple(x): idx for idx, x in enumerate(bbs)}   # theo thu tu truyen vao trong data['shapes']
        rbbs = row_bbs(bbs)
        sorted_bbs = [bb for row in rbbs for bb in row]  # theo thu tu tu trai sang phai, tu tren xuong duoi
        bbs2idx_sorted = {tuple(x): idx for idx, x in enumerate(sorted_bbs)}   # theo thu tu tu trai sang phai, tu tren xuong duoi

        nodes, edges, labels  = [], [], []
        for row_idx, rbb in enumerate(rbbs):
            for bb_idx_in_row, bb in enumerate(rbb):  # duyet qua tung bb (tung node)
                # ----------------- process text feature -----------------
                text = bb2text[bb]
                if self.bpemb.lang != 'vi':
                    text = unidecode.unidecode(text)  # nếu hóa đơn ko dấu thì bpemb để tiếng việt hay tiếng anh ?
                bb_text_feature = get_manual_text_feature(text) + list(np.sum(self.bpemb.embed(text), axis=0))
                text_features.append(bb_text_feature)

                # ----------------- process geometry feature -----------------
                xmin, ymin, xmax, ymax = get_bb_from_poly(bb)
                # augment box coord
                if self.mode == 'train' and np.random.rand() < 0.3:
                    xmin, ymin, xmax, ymax = augment_box(xmin, ymin, xmax, ymax, img_w, img_h, percent=5)

                if self.use_emb: 
                    # rescale coord at width=self.emb_range
                    x_index = [int(xmin * self.emb_range / img_w), int(xmax * self.emb_range / img_w), int((xmax - xmin) * self.emb_range / img_w)]
                    y_index = [int(ymin * self.emb_range / img_h), int(ymax * self.emb_range / img_h), int((ymax - ymin) * self.emb_range / img_h)]
                else:
                    # normalize in rnage(0, 1)
                    x_index = [float(xmin * 1.0 / img_w), float(xmax * 1.0 / img_w), float((xmax - xmin) * 1.0 / img_w)]
                    y_index = [float(ymin * 1.0 / img_h), float(ymax * 1.0 / img_h), float((ymax - ymin) * 1.0 / img_h)]
                x_indexes.append(x_index)
                y_indexes.append(y_index)

                # one hot encode label 
                text_label = bb2label[bb]
                if text_label in self.invalid_labels:
                    text_label = 'text'
                # label = [0] * len(self.label_list)
                # label[self.label_list.index(text_label)] = 1
                # labels.append(label)
                label = self.label_list.index(text_label)
                labels.append(label)

                # add to node   
                nodes.append({
                    'xmin': xmin,
                    'xmax': xmax,
                    'ymin': ymin,
                    'ymax': ymax,
                    'x1': x1,
                    'x2': x2,
                    'x3': x3,
                    'x4': x4,
                    'y1': y1,
                    'y2': y2,
                    'y3': y3,
                    'y4': y4,
                    'feature': [x_index, y_index, bb_text_feature],
                    'label': label
                })
                
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


        return torch.tensor(x_indexes, dtype=torch.int if self.use_emb else torch.float),  \
               torch.tensor(y_indexes, dtype=torch.int if self.use_emb else torch.float), \
               torch.tensor(text_features, dtype=torch.float), \
               edge_index.t().to(torch.int64), \
               edge_type, \
               torch.tensor(labels).type(torch.LongTensor)



class GraphDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setup(stage=None)

    def setup(self, stage):
        self.train_ds = GraphDataset(config=self.config, mode='train')
        self.val_ds = GraphDataset(config=self.config, mode='val')
    
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=None, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=None, shuffle=True, num_workers=8)


if __name__ == '__main__':
    import yaml
    with open("configs/train.yaml") as f:
        general_config = yaml.load(f, Loader=yaml.FullLoader)

    data_module = GraphDataModule(config=general_config)
    pdb.set_trace()

    train_ds = GraphDataset(config=general_config, mode='train')
    for item in train_ds:
        x_indexes, y_indexes, text_features, edge_index, edge_type, labels = item
        for subitem in item:
            print(subitem.shape)
        break












