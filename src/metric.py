def matching_accuracy(gt_dict, pred_dict):
    """
    射影元→射影後の対応におけるMatching Accuracyを計算。

    Parameters:
        gt_dict (dict): 正解の対応関係 {source: target}
        pred_dict (dict): 予測の対応関係 {source: target}

    Returns:
        float: Matching Accuracy（共通射影元のうち射影後が一致している割合）
    """
    matched = 0
    common_keys = set(gt_dict.keys()) & set(pred_dict.keys())

    for key in common_keys:
        if gt_dict[key] == pred_dict[key]:
            matched += 1

    return matched / len(gt_dict) if common_keys else float("nan")


def matching_precision(gt_dict, pred_dict):
    """
    射影後→射影元の対応におけるMatching Precisionを計算。

    Parameters:
        gt_dict (dict): 正解の対応関係 {source: target}
        pred_dict (dict): 予測の対応関係 {source: target}

    Returns:
        float: Matching Precision（予測された射影後のうち正解と一致している割合）
    """
    matched = 0
    common_keys = set(gt_dict.values()) & set(pred_dict.values())

    for key in common_keys:
        if key in pred_dict and pred_dict[key] in gt_dict.values():
            matched += 1

    return matched / len(pred_dict) if pred_dict else float("nan")
