import json
import numpy as np

def iou(boxA, boxB):
    x1, y1 = max(boxA[0], boxB[0]), max(boxA[1], boxB[1])
    x2, y2 = min(boxA[2], boxB[2]), min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def evaluate_standard(file_path, iou_threshold=0.5):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    TP = FP = FN = 0
    iou_list = []

    for item in data:
        gt_all = item.get('ground_truth', {})
        pred_all = item.get('prediction', {})
        classes = set(gt_all) | set(pred_all)  # union of all seen classes
        for cls in classes:
            gts = gt_all.get(cls, [])
            preds = pred_all.get(cls, [])
            matched = set()

            for gt in gts:
                best_iou, best_idx = 0, -1
                for i, pred in enumerate(preds):
                    if i in matched:
                        continue
                    cur_iou = iou(gt, pred)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_idx = i
                if best_iou >= iou_threshold:
                    TP += 1
                    matched.add(best_idx)
                    iou_list.append(best_iou)
                else:
                    FN += 1
            FP += len(preds) - len(matched)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    mean_iou = np.mean(iou_list) if iou_list else 0

    return {
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'Precision@0.5': round(precision, 4),
        'Recall@0.5': round(recall, 4),
        'F1-score@0.5': round(f1_score, 4),
        'Mean IoU@0.5': round(mean_iou, 4)
    }
result = evaluate_standard("./inference/inference4.json")
print(result)