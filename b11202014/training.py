import os, re, glob, math
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import entropy
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import json
import csv

# ---------- helper parsing functions ----------
def parse_number_token(tok):
    """ parse a token that might be a number or percentage string"""
    if isinstance(tok, str):
        tok = tok.strip().replace(',', '')
        if tok.endswith('%'):
            try:
                return float(tok[:-1]) / 100.0
            except:
                return np.nan
        try:
            return float(tok)
        except:
            return np.nan
    return float(tok)

def parse_tokens(line):
    """將一行拆成數字（支援空白或逗號分隔）"""
    if not line:
        return []
    tokens = re.split(r'[\s,]+', line.strip())
    return [parse_number_token(t) for t in tokens if t != '']

def parse_file_to_csv(input_txt, output_csv, debug=False):
    # 讀檔
    with open(input_txt, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]

    all_blocks = []
    i = 0
    n = len(lines)

    # 逐行解析
    while i < n:
        line = lines[i]
        if line.startswith("B[") or line.startswith("W["):  # move line
            res, next_i = parse_move_block(lines, i, debug=debug)
            all_blocks.append(res)
            i = next_i
        else:
            i += 1

    # 把 list 欄位轉字串 (方便寫入 CSV)
    for b in all_blocks:
        for key in ["policy", "value_preds", "rank_out"]:
            if key in b:
                b[key] = json.dumps(b[key], ensure_ascii=False)

    # 寫入 CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_blocks[0].keys())
        writer.writeheader()
        writer.writerows(all_blocks)

    print(f"[INFO] Parsed {len(all_blocks)} move blocks, saved to {output_csv}")

def parse_move_block(lines, i, debug=False):
    """
    解析一手棋的資料區塊
    lines: list of stripped lines
    i: 當前 'B[Q16]' 或 'W[D4]' 的 index
    Return: (res_dict, next_index)
    """
    n = len(lines)
    res = {}

    # move line
    move_line = lines[i].strip()
    res['move'] = move_line
    color = 'B' if move_line.startswith('B[') else ('W' if move_line.startswith('W[') else None)
    res['color'] = color

    if debug:
        print(f"[DEBUG] Parsing move at line {i}: {move_line} (color={color})")

    # collect 接下來的非空行，最多讀 6 行
    vals = []
    j = i + 1
    while j < n and len(vals) < 6:
        if lines[j].strip() != "":
            vals.append(lines[j].strip())
        j += 1

    if debug:
        print(f"[DEBUG] Collected {len(vals)} following lines for analysis")

    # parse 各類欄位
    policy   = parse_tokens(vals[0]) if len(vals) > 0 else []
    value    = parse_tokens(vals[1]) if len(vals) > 1 else []
    rank_out = parse_tokens(vals[2]) if len(vals) > 2 else []

    try:
        strength = parse_number_token(vals[3]) if len(vals) > 3 else np.nan
    except Exception:
        strength = np.nan

    if debug:
        print(f"[DEBUG] policy={len(policy)} vals, value={len(value)} vals, rank_out={len(rank_out)} vals, strength={strength}")

    # KataGo 分析行 (winrate% lead uncertainty)
    winrate, lead, uncert = None, None, None
    if len(vals) > 4:
        kata_tokens = parse_tokens(vals[4])
        if len(kata_tokens) >= 1: winrate = kata_tokens[0]
        if len(kata_tokens) >= 2: lead    = kata_tokens[1]
        if len(kata_tokens) >= 3: uncert  = kata_tokens[2]
        if debug:
            print(f"[DEBUG] KataGo line parsed: winrate={winrate}, lead={lead}, uncert={uncert}")

    # fallback: 沒有 winrate 的話，用 value[0] 當候補
    if winrate is None and len(value) >= 1:
        winrate = value[0]

    # 存到結果
    res['policy']      = policy
    res['value_preds'] = value
    res['rank_out']    = rank_out
    res['strength']    = strength
    res['winrate']     = winrate
    res['lead']        = lead
    res['uncertainty'] = uncert

    if debug:
        print(f"[DEBUG] Finished parsing block, next index={j}")

    return res, j

# ---------- file-level feature extraction ----------
def aggregate_stats(arr, prefix=None):
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        stats = {'mean': np.nan, 'std': np.nan, 'min': np.nan, 'max': np.nan,
                 'median': np.nan, 'p25': np.nan, 'p75': np.nan}
    else:
        stats = {
            'mean': float(np.nanmean(arr)),
            'std': float(np.nanstd(arr)),
            'min': float(np.nanmin(arr)),
            'max': float(np.nanmax(arr)),
            'median': float(np.nanmedian(arr)),
            'p25': float(np.nanpercentile(arr, 25)),
            'p75': float(np.nanpercentile(arr, 75))
        }
    if prefix:
        stats = {f"{prefix}_{k}": v for k, v in stats.items()}
    return stats

def extract_features_from_file(path, save_csv=None, debug=False):
    """
    Parse a file and return a single feature dict (representing the entire file).
    If save_csv exists, load per-move CSV instead of parsing TXT.
    """
    csv_path = save_csv if save_csv else path.replace('.txt', '.csv')

    all_move_dicts = []

    # ----- 先嘗試讀 CSV -----
    if os.path.exists(csv_path):
        if debug:
            print(f"[INFO] CSV exists, loading {csv_path} instead of parsing TXT")
        df_moves = pd.read_csv(csv_path)
        # 將 list 欄位從字串轉回 list
        for key in ['policy', 'value_preds', 'rank_out']:
            if key in df_moves.columns:
                df_moves[key] = df_moves[key].apply(lambda x: json.loads(x) if pd.notna(x) else [])
        all_move_dicts = df_moves.to_dict(orient='records')

    # ----- 若 CSV 不存在，解析 TXT -----
    else:
        if debug:
            print(f"[INFO] Parsing TXT {path} to CSV {csv_path}")
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            raw = f.read().splitlines()
        lines = [ln.strip() for ln in raw if ln.strip() != '']
        i = 0
        while i < len(lines):
            ln = lines[i]
            if ln.startswith('B[') or ln.startswith('W['):
                move_block, next_i = parse_move_block(lines, i, debug=debug)
                i = next_i
                all_move_dicts.append(move_block)
            else:
                i += 1
        # 存 CSV
        if save_csv and len(all_move_dicts) > 0:
            for b in all_move_dicts:
                for key in ['policy', 'value_preds', 'rank_out']:
                    if key in b:
                        b[key] = json.dumps(b[key], ensure_ascii=False)
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=all_move_dicts[0].keys())
                writer.writeheader()
                writer.writerows(all_move_dicts)
            if debug:
                print(f"[INFO] Saved parsed moves to {csv_path}")

    # ====== 從 all_move_dicts 建立特徵 ======
    # per-move collections
    policy_list, value_list, rank_out_list = [], [], []
    strength_list, winrate_list, lead_list, uncert_list = [], [], [], []

    for move_block in all_move_dicts:
        color = move_block.get('color', None)

        pol = move_block.get('policy', [])
        val = move_block.get('value_preds', [])
        ro  = move_block.get('rank_out', [])
        st  = move_block.get('strength', np.nan)
        wr  = move_block.get('winrate', None)
        ld  = move_block.get('lead', None)
        uc  = move_block.get('uncertainty', None)

        if len(pol) > 0: policy_list.append(pol)
        if len(val) > 0: value_list.append(val)
        if len(ro) > 0: rank_out_list.append(ro)
        if not np.isnan(st): strength_list.append(st)

        if wr is not None:
            if color == 'W': wr = 1.0 - wr
            winrate_list.append(wr)
        if ld is not None:
            if color == 'W': ld = -ld
            lead_list.append(ld)
        if uc is not None: uncert_list.append(uc)

    # ====== aggregate features for ML training ======
    feat = {}

    def agg_per_index(list_of_lists, prefix):
        if len(list_of_lists) == 0:
            return {}
        arr = np.array(list_of_lists, dtype=float)
        m = arr.shape[1]
        out = {}
        for idx in range(m):
            stats = aggregate_stats(arr[:, idx])
            for k,v in stats.items():
                out[f'{prefix}_{idx}_{k}'] = v
        out[f'{prefix}_mean_of_idx_means'] = float(np.nanmean(np.nanmean(arr, axis=0)))
        out[f'{prefix}_std_of_idx_means']  = float(np.nanstd(np.nanmean(arr, axis=0)))
        return out

    feat.update(agg_per_index(policy_list, "policy"))
    feat.update(agg_per_index(value_list, "value"))
    feat.update(agg_per_index(rank_out_list, "rank"))

    # 修改 aggregate_stats 接受 array
    feat.update(aggregate_stats(np.array(strength_list)))
    feat.update(aggregate_stats(np.array(winrate_list)))
    feat.update(aggregate_stats(np.array(lead_list)))
    feat.update(aggregate_stats(np.array(uncert_list)))

    return feat

# ---------- dataset builder ----------
def build_dataset_from_train_dir(train_dir):
    print("Building dataset from", train_dir)
    X = []
    y = []
    file_list = []
    # expect files like log_9D_policy_train.txt etc.
    for rank in range(1, 10):
        fname = os.path.join(train_dir, f'log_{rank}D_policy_train.txt')
        if not os.path.exists(fname):
            print("Warning: missing", fname)
            continue
        feats = extract_features_from_file(fname, save_csv=fname.replace('.txt', '.csv'), debug=True)
        X.append(feats)
        y.append(f"{rank}D")
        file_list.append(fname)
    # convert to DataFrame
    X_df = pd.DataFrame(X).fillna(0.0)
    return X_df, np.array(y), file_list

def build_dataset_from_test_files(test_files):
    print("Building dataset from test files")
    X = []
    files = []
    for fpath in test_files:
        feats = extract_features_from_file(fpath)
        X.append(feats)
        files.append(fpath)
    X_df = pd.DataFrame(X).fillna(0.0)
    return X_df, files

# ---------- train/eval ----------
def train_and_evaluate(X_df, y_labels, max_splits=5):
    # label encode
    le = LabelEncoder()
    y_enc = le.fit_transform(y_labels)  # yields 0..8 for 1D..9D

    # 計算每個 class 的樣本數
    counts = Counter(y_labels)
    min_samples_per_class = min(counts.values())

    # 自動調整 n_splits
    n_splits = min(max_splits, min_samples_per_class)
    if n_splits < 2:
        print("[WARN] Not enough samples for cross-validation, skipping CV")
        do_cv = False
    else:
        do_cv = True

    # model
    clf = RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=0,
        n_jobs=-1,
        verbose=1
    )

    # cross-validation
    if do_cv:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
        acc = cross_val_score(clf, X_df.values, y_enc, cv=skf, scoring='accuracy', n_jobs=1)
        print(f"CV accuracy with n_splits={n_splits}: {acc}, mean: {acc.mean():.4f}")
    else:
        print("Skipping cross-validation due to too few samples per class")

    # fit full
    clf.fit(X_df.values, y_enc)
    return clf, le

# ---------- usage example ----------
if __name__ == '__main__':
    train_dir = 'machine-learning-class-fall-2025-assignment-1-q-5/train_set'
    X_df, y_labels, train_files = build_dataset_from_train_dir(train_dir)
    print("Train features shape:", X_df.shape)

    # train model
    clf, le = train_and_evaluate(X_df, y_labels)

    # save model
    joblib.dump((clf, le), 'rank_classifier_rf.joblib')

    # test on test folder
    test_dir = 'machine-learning-class-fall-2025-assignment-1-q-5/test_set'
    test_files = sorted(glob.glob(os.path.join(test_dir, '*.txt')))
    X_test_df, files = build_dataset_from_test_files(test_files)
    preds = clf.predict(X_test_df.values)
    probs = clf.predict_proba(X_test_df.values)

    # 建立 DataFrame
    df_out = pd.DataFrame({
        'id': files,
        'label': le.inverse_transform(preds)
    })

    # 存成 CSV
    df_out.to_csv('predicted_ranks.csv', index=False)
    print("Predictions saved to predicted_ranks.csv")
