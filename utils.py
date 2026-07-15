import math
import numpy as np
import torch
import torch.nn.functional as F


def get_playtime_percentiles_range(dataloader, wr_bucknum, _device):
    all_play_time = []
    for _, (_, label) in enumerate(dataloader):
        all_play_time.append(label)
    all_play_time = torch.cat(all_play_time, dim=0)
    play_time_np = all_play_time.cpu().numpy()
    percen_value = np.percentile(play_time_np, np.linspace(0.0, 100.0, num=wr_bucknum + 1).astype(np.float32)).tolist()
    bucket_begins = torch.tensor(percen_value[:-1], dtype=torch.float32, device=_device).unsqueeze(0)
    bucket_ends = torch.tensor(percen_value[1:], dtype=torch.float32, device=_device).unsqueeze(0)
    return bucket_begins, bucket_ends


def get_tree_classify_loss(label_dict, weight_dict, label_encoding_predict, tree_num_intervals=32):
    auxiliary_loss_ = 0.0
    height = int(math.log2(tree_num_intervals))
    for i in range(height):
        for j in range(2**i):
            interval_label = label_dict[1000*i + j].reshape(-1, 1)
            interval_weight = weight_dict[1000*i + j].reshape(-1, 1)
            interval_preds = label_encoding_predict[:, 2**i - 1 + j].view(-1, 1)
            interval_loss = F.binary_cross_entropy(interval_preds, interval_label, weight=interval_weight)
            auxiliary_loss_ += interval_loss
    return (auxiliary_loss_ / (tree_num_intervals - 1.0)).float()


def get_tree_encoded_label(label, tree_num_intervals, begins, ends):
    label_dict = {}
    weight_dict = {}
    height = int(math.log2(tree_num_intervals))
    for i in range(height):
        for j in range(2**i):
            temp_ind = max(int(tree_num_intervals * 1.0 / (2**i) * j) - 1, 0)
            if j == 0:
                weight_temp = torch.where(label < begins[:, temp_ind].reshape(-1, 1), torch.zeros_like(label), torch.ones_like(label))
            else:
                weight_temp = torch.where(label < ends[:, temp_ind].reshape(-1, 1), torch.zeros_like(label), torch.ones_like(label))
            temp_ind = max(int(tree_num_intervals * 1.0 / (2**i) * (j + 1)) - 1, 0)
            weight_temp = torch.where(label < ends[:, temp_ind].reshape(-1, 1), weight_temp, torch.zeros_like(label))
            temp_ind = max(int(tree_num_intervals * (1.0 / (2**i) * j + 1.0 / (2**(i + 1)))) - 1, 0)
            label_temp = torch.where(label < ends[:, temp_ind].reshape(-1, 1), torch.zeros_like(label), torch.ones_like(label))
            label_dict[1000 * i + j] = label_temp
            weight_dict[1000 * i + j] = weight_temp
    return label_dict, weight_dict


def get_tree_encoded_value(label_encoding_predict, tree_num_intervals, begins, ends):
    height = int(math.log2(tree_num_intervals))
    temp_encoded_playtime = (begins + ends) / 2.0
    batch_size = label_encoding_predict.size(0)
    device = label_encoding_predict.device
    encoded_prob_list = []
    for i in range(tree_num_intervals):
        temp = torch.zeros(batch_size, dtype=torch.float32, device=device)
        cur_code = 2 ** height - 1 + i
        for j in range(1, height + 1):
            classifier_branch = cur_code % 2
            classifier_idx = (cur_code - 1) // 2
            probs = label_encoding_predict[:, classifier_idx]
            condition = torch.tensor(classifier_branch == 1, dtype=torch.bool, device=device)
            log_p = torch.where(condition, torch.log(1.0 - probs + 0.00001), torch.log(probs + 0.00001))
            temp += log_p
            cur_code = classifier_idx
        encoded_prob_list.append(temp)
    encoded_prob = torch.exp(torch.stack(encoded_prob_list, dim=1))
    encoded_playtime = torch.sum(temp_encoded_playtime * encoded_prob, dim=-1, keepdim=True)
    e_x2 = torch.sum((temp_encoded_playtime ** 2) * encoded_prob, dim=-1, keepdim=True)
    var = torch.sqrt(torch.abs(e_x2 - encoded_playtime ** 2) + 1e-8)
    return encoded_playtime.float(), torch.sum(var).float()


class InversePairsCalc:
    def InversePairs(self, data):
        if not data:
            return False
        if len(data) == 1:
            return 0
        def merge(tuple_fir, tuple_sec):
            array_before, cnt_before = tuple_fir
            array_after, cnt_after = tuple_sec
            cnt = cnt_before + cnt_after
            flag = len(array_after) - 1
            array_merge = []
            for i in range(len(array_before) - 1, -1, -1):
                while array_before[i] <= array_after[flag] and flag >= 0:
                    array_merge.append(array_after[flag])
                    flag -= 1
                if flag == -1:
                    break
                else:
                    array_merge.append(array_before[i])
                    cnt += (flag + 1)
            if flag == -1:
                for j in range(i, -1, -1):
                    array_merge.append(array_before[j])
            else:
                for j in range(flag, -1, -1):
                    array_merge.append(array_after[j])
            return array_merge[::-1], cnt
        def mergesort(array):
            if len(array) == 1:
                return (array, 0)
            cut = math.floor(len(array) / 2)
            return merge(mergesort(array[:cut]), mergesort(array[cut:]))
        return mergesort(data)[1]


def eval_mae(labels, scores, scale=1.0):
    return np.mean(np.abs(labels - scores)) * scale


def eval_xauc(labels, pres):
    """Return strict order agreement over all unordered sample pairs.

    Label ties and prediction ties contribute zero. The denominator remains
    N * (N - 1) / 2 to preserve the metric used by the reported tables.
    """
    labels = np.asarray(labels).reshape(-1)
    pres = np.asarray(pres).reshape(-1)
    if labels.shape[0] != pres.shape[0]:
        raise ValueError('labels and predictions must have the same length')
    if not np.isfinite(labels).all() or not np.isfinite(pres).all():
        raise ValueError('labels and predictions must contain only finite values')

    sample_count = labels.shape[0]
    pairs_cnt = sample_count * (sample_count - 1) // 2
    if pairs_cnt == 0:
        return float('nan')

    # Stable sorting is used only to form equal-prediction blocks. Agreements
    # inside each block are subtracted so a prediction tie never receives
    # credit and the result is invariant to the original sample order.
    order = np.argsort(-pres, kind='stable')
    labels_sort = labels[order].tolist()
    preds_sort = pres[order]
    counter = InversePairsCalc()
    total_positive = counter.InversePairs(labels_sort)

    block_start = 0
    while block_start < sample_count:
        block_end = block_start + 1
        while block_end < sample_count and preds_sort[block_end] == preds_sort[block_start]:
            block_end += 1
        if block_end - block_start > 1:
            total_positive -= counter.InversePairs(labels_sort[block_start:block_end])
        block_start = block_end

    return total_positive / pairs_cnt


def eval_by_duration_bucket(labels, scores, durations, play_duration_max=None, dataset_name='kuairec', save_path=None):
    labels    = np.asarray(labels,    dtype=float)
    scores    = np.asarray(scores,    dtype=float)
    durations = np.asarray(durations, dtype=float)

    if dataset_name not in ('kuairec', 'wechat21'):
        return []
    if play_duration_max is None:
        return []

    thresholds_ms = [6_000, 10_000, 18_000, 32_000, 120_000]
    thresholds    = [t / play_duration_max for t in thresholds_ms]
    bucket_labels = ['0-6s', '6-10s', '10-18s', '18-32s', '32-120s', '120s+']
    bucket_ids    = np.digitize(durations, thresholds)

    rows = []
    for b, bname in enumerate(bucket_labels):
        mask = bucket_ids == b
        n = mask.sum()
        if n < 10:
            rows.append({'bucket': bname, 'n': int(n), 'pct': 0.0,
                         'mae': float('nan'), 'xauc': float('nan')})
            continue
        bl, bs = labels[mask], scores[mask]
        rows.append({
            'bucket': bname,
            'n':      int(n),
            'pct':    float(100.0 * n / len(labels)),
            'mae':    eval_mae(bl, bs) * play_duration_max / 1000.0,
            'xauc':   eval_xauc(bl, bs),
        })

    print('\n  ── Duration Bucket Analysis ──────────────────')
    print('  {:10s} {:>8s} {:>6s} {:>10s} {:>8s}'.format('Bucket', 'N', 'Pct%', 'MAE(s)', 'XAUC'))
    print('  ' + '─' * 47)
    for r in rows:
        if np.isnan(r['mae']):
            print('  {:10s} {:>8,d} {:>5.1f}%  {:>10s} {:>8s}'.format(
                  r['bucket'], r['n'], r['pct'], '-', '-'))
        else:
            print('  {:10s} {:>8,d} {:>5.1f}%  {:>10.3f} {:>8.4f}'.format(
                  r['bucket'], r['n'], r['pct'], r['mae'], r['xauc']))
    print('  ' + '─' * 47)

    if save_path is not None and rows:
        import json, os
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({'dataset': dataset_name, 'buckets': rows}, f, indent=2)
    return rows
