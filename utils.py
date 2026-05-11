import math
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn.functional as F
 
import torch
import numpy as np
 
def get_playtime_percentiles_range(dataloader, wr_bucknum, _device):
    all_play_time = []
    for _, (_, label) in enumerate(dataloader):
        play_time = label
        all_play_time.append(play_time)
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
            interval_preds = label_encoding_predict[:, 2**i - 1 + j].view(-1,1)
            # TPM.forward() returns sigmoid probabilities, so use BCE (not BCEwithLogits)
            interval_loss = F.binary_cross_entropy(interval_preds, interval_label, weight=interval_weight)
            auxiliary_loss_ += interval_loss  
    final_loss = auxiliary_loss_ / (tree_num_intervals - 1.0)
    return final_loss.float()
 
def get_tree_encoded_label(label,tree_num_intervals, begins, ends, name="label_encoding"):
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
 
 
def get_tree_encoded_value(label_encoding_predict, tree_num_intervals, begins, ends, name="encoded_playtime"):
    height = int(math.log2(tree_num_intervals))
    encoded_prob_list = []
    
    temp_encoded_playtime = (begins + ends) / 2.0  
    encoded_playtime = temp_encoded_playtime  
    
    batch_size = label_encoding_predict.size(0)
    device = label_encoding_predict.device
 
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
    
    e_x2 = torch.sum((temp_encoded_playtime ** 2) * encoded_prob, dim=-1, keepdim=True)  # H5: use bin midpoints, not scalar mean
    square_of_e_x = encoded_playtime ** 2
    var = torch.sqrt(torch.abs(e_x2 - square_of_e_x) + 1e-8) 
    
    return encoded_playtime.float(), torch.sum(var).float()
 
 
class InversePairsCalc:
    def InversePairs(self, data):
        if not data :
            return False
        if len(data)==1 :
            return 0
        def merge(tuple_fir,tuple_sec):
            array_before = tuple_fir[0]
            cnt_before = tuple_fir[1]
            array_after = tuple_sec[0]
            cnt_after = tuple_sec[1]
            cnt = cnt_before+cnt_after
            flag = len(array_after)-1
            array_merge = []
            for i in range(len(array_before)-1,-1,-1):
                while array_before[i]<=array_after[flag] and flag>=0 :
                    array_merge.append(array_after[flag])
                    flag -= 1
                if flag == -1 :
                    break
                else:
                    array_merge.append(array_before[i])
                    cnt += (flag+1)
            if flag == -1 :
                for j in range(i,-1,-1):
                    array_merge.append(array_before[j])
            else:
                for j in range(flag ,-1,-1):
                    array_merge.append(array_after[j])
            return array_merge[::-1],cnt
 
        def mergesort(array):
            if len(array)==1:
                return (array,0)
            cut = math.floor(len(array)/2)
            tuple_fir=mergesort(array[:cut])
            tuple_sec=mergesort(array[cut:])
            return merge(tuple_fir, tuple_sec)
        return mergesort(data)[1]
 
def eval_xauc(labels, pres):
    label_preds = zip(labels.reshape(-1), pres.reshape(-1))
    sorted_label_preds = sorted(
        label_preds, key=lambda lc: lc[1], reverse=True)
    label_preds_len = len(sorted_label_preds)
    pairs_cnt = label_preds_len * (label_preds_len - 1) / 2
    if pairs_cnt == 0:
        return float('nan')  # undefined for n < 2

    labels_sort = [ele[0] for ele in sorted_label_preds]
    S = InversePairsCalc()
    total_positive = S.InversePairs(labels_sort)
    xauc = total_positive / pairs_cnt
    return xauc

def eval_wxauc(labels, scores):
    labels = np.asarray(labels, dtype=np.float64).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = len(labels)
    if n < 2:
        return float('nan')
    order = np.argsort(labels, kind='stable')
    lb_s = labels[order]; sc_s = scores[order]
    sc_rank = np.argsort(np.argsort(sc_s + np.arange(n) * 1e-14)).tolist()
    BIT = [0] * (n + 2)
    def _upd(p):
        p += 1
        while p <= n: BIT[p] += 1; p += p & (-p)
    def _qry(p):
        if p < 0: return 0
        p += 1; s = 0
        while p > 0: s += BIT[p]; p -= p & (-p)
        return s
    num = 0.0; den = 0.0; seen = 0; i = 0
    while i < n:
        j = i + 1
        while j < n and lb_s[j] == lb_s[i]: j += 1
        w = lb_s[i]
        if seen > 0 and w > 0:
            for k in range(i, j):
                num += w * _qry(sc_rank[k] - 1); den += w * seen
        for k in range(i, j): _upd(sc_rank[k])
        seen += j - i; i = j
    return float(num / den) if den > 0 else float('nan')

def eval_auc(labels, pres):
    from sklearn.metrics import roc_auc_score  # lazy import: only needed for binary tasks
    auc = roc_auc_score(labels, pres)
    return auc

def eval_mae(labels, scores, scale=1.0):
    return np.mean(np.abs(labels - scores)) * scale

def eval_nrmse(labels, scores):
    """Normalized RMSE: RMSE / mean(label). Optimal=0. From TranSUN paper."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mean_label = np.mean(labels)
    if mean_label == 0:
        return float('nan')
    return np.sqrt(np.mean((scores - labels) ** 2)) / mean_label

def eval_nmae(labels, scores):
    """Normalized MAE: MAE / mean(label). Optimal=0. From TranSUN paper."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    mean_label = np.mean(labels)
    if mean_label == 0:
        return float('nan')
    return np.mean(np.abs(scores - labels)) / mean_label

def eval_tre(labels, scores):
    """Total Ratio Error: |sum(y_hat - y) / sum(y_hat)|. Optimal=0."""
    sum_hat = np.sum(scores)
    if sum_hat == 0:
        return float('nan')
    return abs(np.sum(scores - labels) / sum_hat)

def eval_mre(labels, scores):
    """Mean Ratio Error: |(1/M) * sum((y_hat - y) / y_hat)|. Optimal=0."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=float)
    nonzero = scores != 0
    if nonzero.sum() == 0:
        return float('nan')
    return abs(np.mean((scores[nonzero] - labels[nonzero]) / scores[nonzero]))

def eval_by_duration_bucket(labels, scores, durations, play_duration_max=None, dataset_name='kuairec', save_path=None):
    """
    Per-duration-bucket analysis: MAE(s), XAUC, TRE for each video duration bucket.
    KuaiRec / WeChat21: physical thresholds aligned with behavior change points.
    Returns list of dicts and prints ASCII table.
    """
    labels    = np.asarray(labels,    dtype=float)
    scores    = np.asarray(scores,    dtype=float)
    durations = np.asarray(durations, dtype=float)

    if dataset_name not in ('kuairec', 'wechat21'):
        return []   # only meaningful for datasets with real video duration
    if play_duration_max is None:
        print('  [WARN] eval_by_duration_bucket: play_duration_max is None'
              ' for dataset={}, skipping bucket analysis'.format(dataset_name))
        return []

    # Physical thresholds in normalized space (features['duration'] = video_duration_ms / play_duration_max)
    thresholds_ms  = [6_000, 10_000, 18_000, 32_000, 120_000]
    thresholds     = [t / play_duration_max for t in thresholds_ms]
    bucket_labels  = ['0-6s', '6-10s', '10-18s', '18-32s', '32-120s', '120s+']

    bucket_ids = np.digitize(durations, thresholds)   # 0 … 5

    rows = []
    for b, bname in enumerate(bucket_labels):
        mask = bucket_ids == b
        n = mask.sum()
        if n < 10:
            rows.append({'bucket': bname, 'n': int(n), 'pct': 0.0,
                         'mae': float('nan'), 'xauc': float('nan'), 'tre': float('nan')})
            continue
        bl, bs = labels[mask], scores[mask]
        mae_s  = eval_mae(bl, bs) * play_duration_max / 1000.0
        xauc_  = eval_xauc(bl, bs)
        tre_   = eval_tre(bl, bs)
        rows.append({
            'bucket': bname,
            'n':      int(n),
            'pct':    float(100.0 * n / len(labels)),
            'mae':    mae_s,
            'xauc':   xauc_,
            'tre':    tre_,
        })

    # ASCII table
    print('\n  ── Duration Bucket Analysis ──────────────────────────────────')
    print('  {:10s} {:>8s} {:>6s} {:>10s} {:>8s} {:>8s}'.format(
          'Bucket', 'N', 'Pct%', 'MAE(s)', 'XAUC', 'TRE'))
    print('  ' + '─' * 56)
    for r in rows:
        if np.isnan(r['mae']):
            print('  {:10s} {:>8,d} {:>5.1f}%  {:>10s} {:>8s} {:>8s}'.format(
                  r['bucket'], r['n'], r['pct'], '-', '-', '-'))
        else:
            print('  {:10s} {:>8,d} {:>5.1f}%  {:>10.3f} {:>8.4f} {:>8.4f}'.format(
                  r['bucket'], r['n'], r['pct'], r['mae'], r['xauc'], r['tre']))
    print('  ' + '─' * 56)

    if save_path is not None and rows:
        import json, os
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({'dataset': dataset_name, 'buckets': rows}, f, indent=2)
        print('  [bucket json saved → {}]'.format(save_path))
    return rows


def eval_kl(samples_p, samples_q, bins=100, epsilon=1e-10):
    samples_p = np.asarray(samples_p, dtype=float)
    samples_q = np.asarray(samples_q, dtype=float)
    valid = np.isfinite(samples_p) & np.isfinite(samples_q)
    if valid.sum() < 2:
        return float('nan')
    samples_p = samples_p[valid]
    samples_q = samples_q[valid]
    all_data = np.concatenate([samples_p, samples_q])
    range_ = (all_data.min(), all_data.max())
    hist_p, bin_edges = np.histogram(samples_p, bins=bins, density=True, range=range_)
    hist_q, _ = np.histogram(samples_q, bins=bin_edges, density=True)
    bin_width = np.diff(bin_edges)
    hist_p = hist_p * bin_width
    hist_q = hist_q * bin_width
    hist_p = np.clip(hist_p, epsilon, None)
    hist_q = np.clip(hist_q, epsilon, None)
    kl = np.sum(hist_p * np.log(hist_p / hist_q))
    return kl
 