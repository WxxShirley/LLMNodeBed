import csv 
import sys
sys.path.append("../..")
from common import compute_acc_and_f1, UNKNOW


def simiple_evaluate(file_path, valid_label):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)

        pred_list, gt_list, hall_number = [], [], 0
        for line in reader:
            cur_pred, cur_gt = line[1], line[2]
            if cur_pred not in valid_label:
                cur_pred = UNKNOW
                hall_number += 1
            pred_list.append(cur_pred)
            gt_list.append(cur_gt)
        
        acc, macro_f1, _ = compute_acc_and_f1(pred_list, gt_list)
        hallucination = round(hall_number / (len(pred_list) + 1e-9) * 100.0, 2)
        return acc, macro_f1, hallucination
