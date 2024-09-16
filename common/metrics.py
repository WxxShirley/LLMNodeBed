from sklearn.metrics import accuracy_score, f1_score


def compute_acc_and_f1(pred, ground_truth):
    accuracy = accuracy_score(ground_truth, pred) * 100.0
    f1 = f1_score(ground_truth, pred, average="macro") * 100.0
    
    return round(accuracy, 3), round(f1, 3)
