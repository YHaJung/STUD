import json
from utils.file_processing import load_file

def compute_iou(bbox1, bbox2):
    intersection_x_length = max(min(float(bbox1['xbr']), float(bbox2['xbr'])) - max(float(bbox1['xtl']), float(bbox2['xtl'])), 0)
    intersection_y_length = max(min(float(bbox1['ybr']), float(bbox2['ybr'])) - max(float(bbox1['ytl']), float(bbox2['ytl'])), 0)

    area1 = (float(bbox1['xbr']) - float(bbox1['xtl']))*(float(bbox1['ybr']) - float(bbox1['ytl']))
    area2 = (float(bbox2['xbr']) - float(bbox2['xtl']))*(float(bbox2['ybr']) - float(bbox2['ytl']))
    intersect_area = intersection_x_length*intersection_y_length
    whole_area = area1 + area2 - intersect_area
    
    iou = intersect_area / whole_area
    return iou

def calc_iou_performance(pred_infos, ann_infos, ood_threshold=None):
    img_ids = set(ann_infos.keys()).union(pred_infos.keys())
    TP_per_imgs = [0] * len(img_ids)
    True_per_imgs = [0] * len(img_ids) # num of annotation true
    Pos_per_imgs = [0] * len(img_ids) # num of prediction true

    for img_idx, img_id in enumerate(img_ids):
        if img_id in pred_infos.keys():
            if pred_infos[img_id]['bboxes'] == None:
                pred_infos[img_id]['bboxes'] = []
            if ood_threshold != None:
                pred_infos[img_id]['bboxes'] = [bbox for bbox in pred_infos[img_id]['bboxes'] if (bbox['label'] != 2) and (bbox['ood_score'] > ood_threshold)]
            else:
                pred_infos[img_id]['bboxes'] = [bbox for bbox in pred_infos[img_id]['bboxes'] if bbox['label'] == 0]
        True_per_imgs[img_idx] = len(ann_infos[img_id]['bboxes']) if img_id in ann_infos.keys() else 0
        Pos_per_imgs[img_idx] = len(pred_infos[img_id]['bboxes']) if img_id in pred_infos.keys() else 0

        if True_per_imgs[img_idx] != 0 and Pos_per_imgs[img_idx] != 0:
            for ann_bbox in ann_infos[img_id]['bboxes']:
                for pred_bbox in pred_infos[img_id]['bboxes']:
                    iou = compute_iou(ann_bbox, pred_bbox)
                    if ood_threshold:
                        pred_label = 0 if pred_bbox['ood_score'] > ood_threshold else 1
                    else:
                        pred_label = pred_bbox['label'] if 'label' in pred_bbox.keys() else 0
                    if (iou >= 0.5) and (pred_label == ann_bbox['label']):
                        TP_per_imgs[img_idx] += 1
                        break

    precision = round(sum(TP_per_imgs) / (sum(Pos_per_imgs) + 1e-15), 4)
    recall = round(sum(TP_per_imgs) / (sum(True_per_imgs) + 1e-15), 4)
    f1_score = round((2 * precision * recall) / (precision + recall + 1E-15), 4)

    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


if __name__=='__main__':
    ann_path = 'datasets/custom102/annotations/all.json'
    pred_path = 'datasets/custom102/yolov7_preds/yolov7_predictions.json'

    with open(ann_path, "r") as json_file:
        ann_info = json.load(json_file)
    with open(pred_path, "r") as json_file:
        pred_info = json.load(json_file)
    
    print(calc_iou_performance(pred_info, ann_info, conf_threshold=0.05))