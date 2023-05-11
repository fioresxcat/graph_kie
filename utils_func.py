import os
import numpy as np
import json
from pathlib import Path
import shutil
from pprint import pprint
from my_utils import *
from PIL import Image
import pdb
import cv2


def get_img_fp_from_json_fp(json_fp: Path):
    ls_ext = ['.jpg', '.png', '.jpeg', '.JPG', '.PNG', '.JPEG']
    for ext in ls_ext:
        img_fp = json_fp.with_suffix(ext)
        if img_fp.exists():
            return img_fp
    return None


def split_val_test(val_dir, new_test_dir, keep_ratio=0.5):
    os.makedirs(new_test_dir, exist_ok=True)

    ls_jp = list(Path(val_dir).glob('*.json'))
    print(f'src dir has {len(ls_jp)} json files')
    np.random.shuffle(ls_jp)
    cnt = 0
    num_move = int((1 - keep_ratio) * len(ls_jp))
    for jp in ls_jp[:num_move]:
        shutil.move(str(jp), new_test_dir)
        img_fp = get_img_fp_from_json_fp(jp)
        if img_fp:
            shutil.move(str(img_fp), new_test_dir)
            cnt += 1
    print(f'moved {cnt} file to {new_test_dir}')


def check_json_and_img(dir, del_if_not_exist=False):
    ls_jp = list(Path(dir).glob('*.json'))
    for jp in ls_jp:
        img_fp = get_img_fp_from_json_fp(jp)
        if img_fp is None:
            print(f'{jp} does not have corresponding image')
            if del_if_not_exist:
                os.remove(jp)
                print(f'removed {jp}')


def check_number_of_data(root_dir):
    res = {}
    for mart_type in os.listdir(root_dir):
        res[mart_type] = {}
        for split_dir in Path(os.path.join(root_dir, mart_type)).glob('*'):
            res[mart_type][split_dir.name] = len(list(split_dir.rglob('*.json')))

    print(json.dumps(res, indent=4))


def remove_duplicate_in_split(data_dir):
    ls_train_fp = list(Path(os.path.join(data_dir, 'train_drop_unit_price')).rglob('*.json'))

    ls_val_fp = list(Path(os.path.join(data_dir, 'val_drop_unit_price')).rglob('*.json'))
    ls_val_fn = [fp.name for fp in ls_val_fp]
    ls_test_fp = list(Path(os.path.join(data_dir, 'test_drop_unit_price')).rglob('*.json'))
    ls_test_fn = [fp.name for fp in ls_test_fp]

    cnt = 0
    for fp in ls_train_fp:
        fp2del = None
        if fp.name in ls_val_fn:
            fp2del = ls_val_fp[ls_val_fn.index(fp.name)]
        if fp.name in ls_test_fn:
            fp2del = ls_test_fp[ls_test_fn.index(fp.name)]
        if fp2del is not None:
            img_fp2del = get_img_fp_from_json_fp(fp2del)
            os.remove(fp2del)
            os.remove(img_fp2del)
            print(f'remove {fp2del}')
            cnt += 1
    print(f'remove {cnt} duplicate files!')


def remove_duplicate_bb(json_fp):
    json_data = json.load(open(json_fp))
    ls_idx2del = []
    bbs = []
    for i, shape in enumerate(json_data['shapes']):
        bb = [coord for pt in shape['points'] for coord in pt]
        if bb not in bbs:
            bbs.append(bb)
        else:
            ls_idx2del.append(i)
    if len(ls_idx2del) > 0:
        json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in ls_idx2del]
        with open(json_fp, 'w') as f:
            json.dump(json_data, f)
        print(f'{json_fp} has {len(ls_idx2del)} duplicate boxes')
    
    
def check_all_coord_valid(data_dir, del_invalid_poly=False):
    """
        Check
          + if there is any poly with different than 4 points
          + if there is any poly with coord smaller than 0 or bigger than image size (correct luôn)
          + if there is any poly not ocred
    """
    ls_json_fp = list(Path(data_dir).rglob('*.json'))
    ls_json_fp = [fp for fp in ls_json_fp if get_img_fp_from_json_fp(fp) is not None]
    for jp in ls_json_fp:
        json_data = json.load(open(jp))
        img_fp = get_img_fp_from_json_fp(jp)
        img_w, img_h = Image.open(img_fp).size
        ls_idx2del = []
        is_fixed = False
        for i, shape in enumerate(json_data['shapes']):
            if 'text' not in shape:
                # print(f'{jp} has poly not ocred')
                # print(shape)
                if del_invalid_poly:
                    ls_idx2del.append(i)

            if len(shape['points']) == 4:
                x1, y1, x2, y2, x3, y3, x4, y4 = [coord for pt in shape['points'] for coord in pt]
                x1 = max(0, min(x1, img_w))
                x2 = max(0, min(x2, img_w))
                x3 = max(0, min(x3, img_w))
                x4 = max(0, min(x4, img_w))
                y1 = max(0, min(y1, img_h))
                y2 = max(0, min(y2, img_h))
                y3 = max(0, min(y3, img_h))
                y4 = max(0, min(y4, img_h))
                
                json_data['shapes'][i]['points'] = [
                    [x1, y1], [x2, y2], [x3, y3], [x4, y4]
                ]
            else:
                if del_invalid_poly:
                    ls_idx2del.append(i)
                n_points = len(shape["points"])
                # print(f'{jp} has poly with {n_points} points')
                if n_points == 5:  # fix
                    n_invalid_pts = 0
                    for pt in shape['points']:
                        if any([x-int(x)>0.01 for x in pt]):
                            n_invalid_pts += 1
                    json_data['shapes'][i]['points'] = [pt for pt in shape['points'] if all([x-int(x)<0.01 for x in pt])]
                    print(f'fixed 5 points shape in {jp}')
                    is_fixed = True
                    # print(f'{jp} has poly with 5 points and {n_invalid_pts} invalid points')
                    # print(shape)
                elif n_points == 2 and shape['shape_type'] == 'rectangle': # (xmin, ymin, xmax, ymax)
                    xmin, ymin, xmax, ymax = [coord for pt in shape['points'] for coord in pt]
                    json_data['shapes'][i]['points'] = [
                        [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]
                    ]
                    print(f'fixed rectangle shape in {jp}')
                    is_fixed = True
            
            if shape['shape_type'] == 'rectangle':
                json_data['shapes'][i]['shape_type'] = 'polygon'
                print(f'fixed rectangle shape in {jp}')
                is_fixed = True


        json_data['shapes'] = [shape for i, shape in enumerate(json_data['shapes']) if i not in ls_idx2del]
        if is_fixed:
            with open(jp, 'w') as f:
                json.dump(json_data, f)
        # print(f'Done checking {jp}')




if __name__ == '__main__':
    # split_val_test(
    #     val_dir='../ie_data/winmart_combined/val/winmart_2022_val_out',
    #     new_test_dir='../ie_data/winmart_combined/test/winmart_2022_val_out',
    #     keep_ratio=0.5
    # )

    # check_number_of_data('../ie_data')

    # for mart_type in os.listdir('../ie_data'):
    #     print(f'Processing {mart_type}')
    #     data_dir = os.path.join('../ie_data', mart_type)
    #     remove_duplicate_in_split(data_dir)

    # res = {}
    # for martname, data in DATA_DICT.items():
    #     res[martname] = {}
    #     res[martname][data['train_dir']] = len(list(Path(data['train_dir']).rglob('*.json')))
    #     res[martname][data['val_dir']] = len(list(Path(data['val_dir']).rglob('*.json')))
    # print(json.dumps(res, indent=4))

    # root_dir = '../ie_data'
    # ls_jp = list(Path(root_dir).rglob('*.json'))
    # for jp in ls_jp:
    #     # print(jp)
    #     remove_duplicate_bb(json_fp=jp)
    #     # break

    root_dir = '../ie_data'
    check_all_coord_valid(root_dir, del_invalid_poly=False)
