import os
import re
import pdb
import json
import math
import numpy as np
import unidecode
from bpemb import BPEmb
from pathlib import Path
from typing import List, Tuple
import copy
from model import *

DATA_DICT = {
    'aeon_citimart': {
        'train_dir': '../ie_data/aeon_citimart/train',
        'val_dir': '../ie_data/aeon_citimart/val',
        'test_dir': '../ie_data/aeon_citimart/test',
    },
    'aeon_combined': {
        'train_dir': '../ie_data/aeon_combined/train',
        'val_dir': '../ie_data/aeon_combined/val',
        'test_dir': '../ie_data/aeon_combined/test',

    },
    'brg': {
        'train_dir': '../ie_data/brg/train',
        'val_dir': '../ie_data/brg/val',
        'test_dir': '../ie_data/brg/test',

    },
    'coopmart_combined_new_out_2': {
        'train_dir': '../ie_data/coopmart_combined_new_out_2/train_drop_unit_price',
        'val_dir': '../ie_data/coopmart_combined_new_out_2/val_drop_unit_price',
        'test_dir': '../ie_data/coopmart_combined_new_out_2/test_drop_unit_price',

    },
    'emart': {
        'train_dir': '../ie_data/emart/train',
        'val_dir': '../ie_data/emart/val',
        'test_dir': '../ie_data/emart/test',

    },
    'fujimart': {
        'train_dir': '../ie_data/fujimart/train',
        'val_dir': '../ie_data/fujimart/val',
        'test_dir': '../ie_data/fujimart/test',

    },
    'lotte-drop-0.4': {
        'train_dir': '../ie_data/lotte-drop-0.4/train',
        'val_dir': '../ie_data/lotte-drop-0.4/val',
        'test_dir': '../ie_data/lotte-drop-0.4/test',

    },
    'satra': {
        'train_dir': '../ie_data/satra/train',
        'val_dir': '../ie_data/satra/val',
        'test_dir': '../ie_data/satra/test',

    },
    'tgs': {
        'train_dir': '../ie_data/tgs/train',
        'val_dir': '../ie_data/tgs/val',
        'test_dir': '../ie_data/tgs/test',

    },
    'winmart_combined': {
        'train_dir': '../ie_data/winmart_combined/train',
        'val_dir': '../ie_data/winmart_combined/val',
        'test_dir': '../ie_data/winmart_combined/test'

    },
    'bigc_old': {
        'train_dir': '../ie_data/bigc_old/train',
        'val_dir': '../ie_data/bigc_old/val',
        'test_dir': '../ie_data/bigc_old/test',

    },
    'mega_2022': {
        'train_dir': '../ie_data/mega_2022/train',
        'val_dir': '../ie_data/mega_2022/val',
        'test_dir': '../ie_data/mega_2022/test'

    },
}


uc = {
    'a':'a',
    'á':'a',
    'à':'a',
    'ả':'a',
    'ã':'a',
    'ạ':'a',
    'ă':'a',
    'ắ':'a',
    'ằ':'a',
    'ẳ':'a',
    'ẵ':'a',
    'ặ':'a',
    'â':'a',
    'ấ':'a',
    'ầ':'a',
    'ẩ':'a',
    'ẫ':'a',
    'ậ':'a',
    'e':'e',
    'é':'e',
    'è':'e',
    'ẻ':'e',
    'ẽ':'e',
    'ẹ':'e',
    'ê':'e',
    'ế':'e',
    'ề':'e',
    'ể':'e',
    'ễ':'e',
    'ệ':'e',
    'i':'i',
    'í':'i',
    'ì':'i',
    'ỉ':'i',
    'ĩ':'i',
    'ị':'i',
    'o':'o',
    'ó':'o',
    'ò':'o',
    'ỏ':'o',
    'õ':'o',
    'ọ':'o',
    'ô':'o',
    'ố':'o',
    'ồ':'o',
    'ổ':'o',
    'ỗ':'o',
    'ộ':'o',
    'ơ':'o',
    'ớ':'o',
    'ờ':'o',
    'ở':'o',
    'ỡ':'o',
    'ợ':'o',
    'u':'u',
    'ú':'u',
    'ù':'u',
    'ủ':'u',
    'ũ':'u',
    'ụ':'u',
    'ư':'u',
    'ứ':'u',
    'ừ':'u',
    'ử':'u',
    'ữ':'u',
    'ự':'u',
    'y':'y',
    'ý':'y',
    'ỳ':'y',
    'ỷ':'y',
    'ỹ':'y',
    'ỵ':'y',
    'đ':'d'
}


SUPPORTED_MODEL = {
    'rgcn': RGCN_Model,
    'gatv2': GATv2_Model,
    'gnn_film': GNN_FiLM_Model
}


def load_model(general_cfg, model_cfg, n_classes, ckpt_path=None):
    model_type = general_cfg['options']['model_type']
    if model_type not in SUPPORTED_MODEL:
        raise ValueError(f'Model type {model_type} is not supported yet')
    
    if ckpt_path is not None:
        model = SUPPORTED_MODEL[model_type].load_from_checkpoint(
                    checkpoint_path=ckpt_path,
                    general_cfg=general_cfg, 
                    model_cfg=model_cfg, 
                    n_classes=n_classes
                )
    else:
        model = SUPPORTED_MODEL[model_type](
            general_cfg=general_cfg, 
            model_cfg=model_cfg, 
            n_classes=n_classes
        )
    
    return model



def remove_accent(text):
    return unidecode.unidecode(text)


def rotate(xy, theta):
    # https://en.wikipedia.org/wiki/Rotation_matrix#In_two_dimensions
    cos_theta, sin_theta = math.cos(theta), math.sin(theta)

    return (
        xy[0] * cos_theta - xy[1] * sin_theta,
        xy[0] * sin_theta + xy[1] * cos_theta
    )


def translate(xy, offset, img_h):
    return [xy[0] + offset[0], img_h - (xy[1] + offset[1])]

def max_left(bb):
    return min(bb[0], bb[2], bb[4], bb[6])

def max_right(bb):
    return max(bb[0], bb[2], bb[4], bb[6])

def row_bbs(bbs):
    bbs.sort(key=lambda x: max_left(x))
    clusters, y_min = [], []
    for tgt_node in bbs:
        if len (clusters) == 0:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
            continue
        matched = None
        tgt_7_1 = tgt_node[7] - tgt_node[1]
        min_tgt_0_6 = min(tgt_node[0], tgt_node[6])
        max_tgt_2_4 = max(tgt_node[2], tgt_node[4])
        max_left_tgt = max_left(tgt_node)
        for idx, clt in enumerate(clusters):
            src_node = clt[-1]
            src_5_3 = src_node[5] - src_node[3]
            max_src_2_4 = max(src_node[2], src_node[4])
            min_src_0_6 = min(src_node[0], src_node[6])
            overlap_y = (src_5_3 + tgt_7_1) - (max(src_node[5], tgt_node[7]) - min(src_node[3], tgt_node[1]))
            overlap_x = (max_src_2_4 - min_src_0_6) + (max_tgt_2_4 - min_tgt_0_6) - (max(max_src_2_4, max_tgt_2_4) - min(min_src_0_6, min_tgt_0_6))
            if overlap_y > 0.5*min(src_5_3, tgt_7_1) and overlap_x < 0.6*min(max_src_2_4 - min_src_0_6, max_tgt_2_4 - min_tgt_0_6):
                distance = max_left_tgt - max_right(src_node)
                if matched is None or distance < matched[1]:
                    matched = (idx, distance)
        if matched is None:
            clusters.append([tgt_node])
            y_min.append(tgt_node[1])
        else:
            idx = matched[0]
            clusters[idx].append(tgt_node)
    zip_clusters = list(zip(clusters, y_min))
    zip_clusters.sort(key=lambda x: x[1])
    zip_clusters = list(np.array(zip_clusters, dtype=object)[:, 0])
    return zip_clusters

def sort_bbs(bbs):
    bb_clusters = row_bbs(bbs)
    bbs = []
    for cl in bb_clusters:
        bbs.extend(cl)
    return bbs, bb_clusters


def sort_json(json_data):
    bbs, labels, texts = [], [], []
    for shape in json_data['shapes']:
        x1, y1 = shape['points'][0]  # tl
        x2, y2 = shape['points'][1]  # tr
        x3, y3 = shape['points'][2]  # br
        x4, y4 = shape['points'][3]  # bl
        bb = tuple(int(i) for i in (x1,y1,x2,y2,x3,y3,x4,y4))
        bbs.append(bb)
        labels.append(shape['label'])
        texts.append(shape['text'])

    bb2label = dict(zip(bbs, labels))   # theo thu tu truyen vao trong data['shapes']
    bb2text = dict(zip(bbs, texts))
    bb2idx_original = {x: idx for idx, x in enumerate(bbs)}   # theo thu tu truyen vao trong data['shapes']
    rbbs = row_bbs(copy.deepcopy(bbs))
    sorted_bbs = [bb for row in rbbs for bb in row]  # theo thu tu tu trai sang phai, tu tren xuong duoi
    bb2idx_sorted = {tuple(x): idx for idx, x in enumerate(sorted_bbs)}   # theo thu tu tu trai sang phai, tu tren xuong duoi
    sorted_indices = [bb2idx_sorted[bb] for bb in bb2idx_original.keys()]

    return bb2label, bb2text, rbbs, bb2idx_sorted, sorted_indices



def unsign(text):
    unsign_text = ''
    for c in text.lower():
        if c in uc.keys():
            unsign_text += uc[c]
        else:
            unsign_text += c
    return unsign_text


def get_img_fp_from_json_fp(json_fp: Path):
    ls_ext = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in ls_ext:
        img_fp = json_fp.with_suffix(ext)
        if img_fp.exists():
            return img_fp
    return None


def augment_text(text):
    """
        randomly change number in the original text
    """
    augmented_text = ''
    for c in text:
        if c.isdigit():
            augmented_text += str(np.random.randint(0, 10))
        else:
            augmented_text += c
    
    return augmented_text


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



def get_list_json(data_dir: str, max_sample=int(1e4), remove_json_with_no_img=True, shuffle=False):
    ls_json_fp = sorted(list(Path(data_dir).rglob('*.json')))
    if remove_json_with_no_img:
        ls_json_fp = [fp for fp in ls_json_fp if get_img_fp_from_json_fp(fp) is not None]
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
            ls_idx2drop = np.random.choice(non_text_indices, min(num_field_drop, len(non_text_indices)//8))
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


def get_word_encoder(encoder_options):
    encoder = BPEmb(
        lang=encoder_options['lang'],
        dim=encoder_options['dim'],
        vs=encoder_options['vs']
    )
    return encoder


def get_experiment_dir(root_dir, description=None):
    os.makedirs(root_dir, exist_ok=True)
    exp_nums = [int(subdir[3:]) if '_' not in subdir else int(subdir.split('_')[0][3:]) for subdir in os.listdir(root_dir)]
    max_exp_num = max(exp_nums) if len(exp_nums) > 0 else 0
    exp_name = f'exp{max_exp_num+1}' if description is None else f'exp{max_exp_num+1}_{description}'
    return os.path.join(root_dir, exp_name)
