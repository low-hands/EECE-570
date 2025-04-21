import json
import numpy as np

def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

# Load inference data
with open('./eval/inference4.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Gather all classes
classes = set()
for item in data:
    classes.update(item.get('ground_truth', {}).keys())
    classes.update(item.get('prediction', {}).keys())
classes = sorted(classes)

# IoU thresholds for mAP
thresholds = np.arange(0.5, 1.0, 0.05)

map_scores = []
iou_list = []      # For mean IoU
precisions = []    # Precision per class @0.5
recalls = []       # Recall per class @0.5

for t in thresholds:
    precs = []
    recs = []
    for cls in classes:
        tp = fp = fn = 0
        cls_ious = []
        for item in data:
            gt_boxes = item.get('ground_truth', {}).get(cls, [])
            pred_boxes = item.get('prediction', {}).get(cls, [])
            matched = set()
            for pb in pred_boxes:
                best_iou = 0
                best_idx = -1
                for idx, gb in enumerate(gt_boxes):
                    if idx in matched:
                        continue
                    cur_iou = iou(pb, gb)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_idx = idx
                if best_iou >= t:
                    tp += 1
                    matched.add(best_idx)
                    cls_ious.append(best_iou)
                else:
                    fp += 1
            fn += len(gt_boxes) - len(matched)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        precs.append(prec)
        recs.append(rec)
        if t == 0.5:
            iou_list.extend(cls_ious)
    map_scores.append(np.mean(precs))
    if t == 0.5:
        precisions = precs
        recalls = recs

# Final Metrics
map50 = map_scores[0]
map50_95 = np.mean(map_scores)
mean_iou = np.mean(iou_list) if iou_list else 0
mean_prec = np.mean(precisions)
mean_recall = np.mean(recalls)
f1_score = (2 * mean_prec * mean_recall) / (mean_prec + mean_recall) if (mean_prec + mean_recall) > 0 else 0

# Display metrics in table order
print(f"mAP@50: {map50:.4f}")
print(f"mAP@50:0.95: {map50_95:.4f}")
print(f"Precision@0.5: {mean_prec:.4f}")
print(f"Recall@0.5: {mean_recall:.4f}")
print(f"F1-score@0.5: {f1_score:.4f}")
print(f"Mean IoU: {mean_iou:.4f}")
