"""
Lynch症候群 費用対効果分析 — 計算エンジン (cea_core.py)
計算ロジックのみ。Streamlit (st) への依存なし。

【Markovモデル状態（マルコフ.jpg 新モデル対応）】
  healthy       : Healthy          (効用値乗数 1.0)
  adenoma       : Adenoma(内視鏡)  (効用値乗数 0.99)
  early         : Early(内視鏡)    (効用値乗数 0.95)
  local         : Local(手術→化療)  (効用値乗数 0.7)
  dfs_l         : DFS_L            (効用値乗数 0.75)  ← 旧 dfs_m
  metastatic    : Metastatic(化療)  (効用値乗数 0.6)   ← 旧 advanced
  dfs_m         : DFS_M            (効用値乗数 0.75)  ← 旧 dfs_a
  bsc           : BSC              (効用値乗数 0.25)
  death         : Death            (効用値 0.0)
"""

import warnings
from copy import deepcopy

import numpy as np
import pandas as pd

MORTALITY_TABLE_2024 = {
    0: 0.001820, 1: 0.000250, 2: 0.000185, 3: 0.000135, 4: 0.000105,
    5: 0.000085, 6: 0.000080, 7: 0.000070, 8: 0.000065, 9: 0.000065,
    10: 0.000065, 11: 0.000075, 12: 0.000095, 13: 0.000115, 14: 0.000150,
    15: 0.000195, 16: 0.000240, 17: 0.000280, 18: 0.000315, 19: 0.000360,
    20: 0.000395, 21: 0.000420, 22: 0.000445, 23: 0.000465, 24: 0.000490,
    25: 0.000505, 26: 0.000525, 27: 0.000540, 28: 0.000560, 29: 0.000580,
    30: 0.000600, 31: 0.000630, 32: 0.000655, 33: 0.000685, 34: 0.000720,
    35: 0.000760, 36: 0.000805, 37: 0.000855, 38: 0.000915, 39: 0.000985,
    40: 0.001060, 41: 0.001140, 42: 0.001230, 43: 0.001330, 44: 0.001445,
    45: 0.001565, 46: 0.001700, 47: 0.001845, 48: 0.002005, 49: 0.002190,
    50: 0.002390, 51: 0.002610, 52: 0.002835, 53: 0.003070, 54: 0.003310,
    55: 0.003550, 56: 0.003780, 57: 0.004005, 58: 0.004225, 59: 0.004450,
    60: 0.004680, 61: 0.004925, 62: 0.005185, 63: 0.005470, 64: 0.005775,
    65: 0.006155, 66: 0.006615, 67: 0.007120, 68: 0.007685, 69: 0.008330,
    70: 0.009040, 71: 0.009845, 72: 0.010775, 73: 0.011855, 74: 0.013110,
    75: 0.014525, 76: 0.016080, 77: 0.017775, 78: 0.019625, 79: 0.021705,
    80: 0.024035, 81: 0.026670, 82: 0.029670, 83: 0.033110, 84: 0.037045,
    85: 0.041635, 86: 0.047055, 87: 0.053150, 88: 0.059875, 89: 0.067305,
    90: 0.075450, 91: 0.084345, 92: 0.093905, 93: 0.104050, 94: 0.114755,
    95: 0.126130, 96: 0.138130, 97: 0.150535, 98: 0.163485, 99: 0.177095,
    100: 0.191265, 101: 0.205735, 102: 0.221535, 103: 0.239335, 104: 0.260020,
    105: 1.000000
}

def get_background_mortality(age):
    return MORTALITY_TABLE_2024.get(int(age), 1.0)

COEFF_3 = -1.93265216e-06
COEFF_2 = 2.46095009e-04
COEFF_1 = -1.04300519e-02
INTERCEPT = 1.09016546

def get_utility_norm(age):
    util = (COEFF_3 * (age**3)) + (COEFF_2 * (age**2)) + (COEFF_1 * age) + INTERCEPT
    return float(np.clip(util, 0.0, 1.0))

def survival_to_annual(t, S):
    """X[t, S] = 1-(1-S)^(1/t): t年累積遷移率SからMarkov年次遷移確率へ変換

    Markovモデル図（マルコフ.jpg）の表記 X[t, S(t)] = 1-(1-S(t))^(1/t) に対応。

    S: t年間の累積遷移率（イベント発生率）。
    境界条件:
      S = 0 → 年次遷移確率 = 0.0
      S = 1 → 年次遷移確率 = 1.0
    """
    if S <= 0:
        return 0.0
    if S >= 1:
        return 1.0
    return 1.0 - ((1.0 - S) ** (1.0 / t))

def _normalize(probs: dict) -> dict:
    """
    遷移確率の辞書を受け取り、常に合計=1.0 に正規化して返す。
    各値は非負にクリップしたうえで正規化する。
    合計がゼロの場合は均等分配する（フォールバック）。
    """
    clipped = {k: max(0.0, v) for k, v in probs.items()}
    total = sum(clipped.values())
    if total <= 0.0:
        n = len(clipped)
        return {k: 1.0 / n for k in clipped}
    return {k: v / total for k, v in clipped.items()}


def compute_transition_probs(params, has_surveillance=True):
    """
    Markovモデルの disease-specific 遷移確率を計算。
    has_surveillance=True/False でサーベイランス有無を切り替え。

    【新Markovモデル（マルコフ.jpg）の遷移構造】
    Healthy self-loop: 1 - X[10,0.6] - X[10,0.2] - X[10,0.08] - X[10,0.01] - 他因死
    → Healthy から Adenoma/Early/Local/Metastatic へ直接遷移あり

    Healthy:
      Healthy → Adenoma:    X[10, 0.6]  (サーベイランスあり:surv / なし:nosurv で別設定)
      Healthy → Early:      X[10, 0.2]
      Healthy → Local:      X[10, 0.08]
      Healthy → Metastatic: X[10, 0.01]

    Adenoma (内視鏡):
      Adenoma → Healthy (完全除去): 1.0

    Early (Endoscopy):
      Early → Healthy (治癒): 1.0
      ※ Early状態からLocalへの直接進行は実装しない（図の構造上、Early=内視鏡で処置完了）

    Local (Surg→Chemo):
      Local → DFS_L:      1.0（直接確率）
      Local → Metastatic: X[10, 0.01]

    DFS_L（5年観察）:
      DFS_L → Healthy:    X[5, 0.7]
      DFS_L → Metastatic: X[5, 0.3]

    Metastatic（化学療法）:
      Metastatic → DFS_M:   X[5, 0.4]
      Metastatic → BSC:     X[5, 0.6]
      Metastatic → Healthy: X[5, 0.5]
      ⚠️ S値の合計 (0.4+0.6+0.5=1.5) が1を超えている点に注意。
      年次確率への変換後 (X[5,S] = 1-(1-S)^0.2) は各々 ≈0.098/0.169/0.129 となり合計≈0.396<1 で
      数学的には問題ないが、各S値が互いに競合リスクの関係にあるため、
      パラメータ設定時は累積競合リスクの整合性を文献値で確認すること。

    DFS_M:
      DFS_M → Healthy:    X[5, 0.5]
      DFS_M → Metastatic: X[5, 0.5]

    BSC:
      BSC → Death: X[1, 0.98]
    """
    p = params

    # ---- Healthy → Adenoma / Early / Local / Metastatic / stay ----
    if has_surveillance:
        r_healthy_to_adn = survival_to_annual(
            p['tp_healthy_adn_surv_t'], p['tp_healthy_adn_surv_s']
        )
    else:
        r_healthy_to_adn = survival_to_annual(
            p['tp_healthy_adn_nosurv_t'], p['tp_healthy_adn_nosurv_s']
        )
    # Healthy → Early / Local / Metastatic:
    # サーベイランスあり: 早期発見が多い（Early↑、Local↓、Meta↓）
    # サーベイランスなし: 進行癌発見が多い（Early↓、Local↑、Meta↑）
    if has_surveillance:
        r_healthy_to_early = survival_to_annual(
            p['tp_healthy_early_t'], p['tp_healthy_early_s']
        )
        r_healthy_to_local = survival_to_annual(
            p['tp_healthy_local_t'], p['tp_healthy_local_s']
        )
        r_healthy_to_meta = survival_to_annual(
            p['tp_healthy_meta_t'], p['tp_healthy_meta_s']
        )
    else:
        r_healthy_to_early = survival_to_annual(
            p['tp_healthy_early_nosurv_t'], p['tp_healthy_early_nosurv_s']
        )
        r_healthy_to_local = survival_to_annual(
            p['tp_healthy_local_nosurv_t'], p['tp_healthy_local_nosurv_s']
        )
        r_healthy_to_meta = survival_to_annual(
            p['tp_healthy_meta_nosurv_t'], p['tp_healthy_meta_nosurv_s']
        )

    tp = {}

    # ---- Healthy → Adenoma / Early / Local / Metastatic / stay ----
    _h_stay = max(0.0, 1.0 - r_healthy_to_adn - r_healthy_to_early
                           - r_healthy_to_local - r_healthy_to_meta)
    h_norm = _normalize({
        'adn':   r_healthy_to_adn,
        'early': r_healthy_to_early,
        'local': r_healthy_to_local,
        'meta':  r_healthy_to_meta,
        'stay':  _h_stay,
    })
    tp['healthy_to_adn']   = h_norm['adn']
    tp['healthy_to_early'] = h_norm['early']
    tp['healthy_to_local'] = h_norm['local']
    tp['healthy_to_meta']  = h_norm['meta']
    tp['healthy_stay']     = h_norm['stay']

    # ---- Adenoma → Healthy / stay ----
    # 図のとおり Adenoma→Healthy=1.0（腺腫は内視鏡で完全除去）
    tp['adn_healthy'] = max(0.0, min(1.0, p['tp_adenoma_healthy']))
    tp['adn_stay']    = max(0.0, 1.0 - tp['adn_healthy'])

    # ---- Early (Endoscopy) → Healthy(治癒) / stay ----
    # 図のとおり Early→Healthy=1.0、Early→Localなし
    r_early_cure = p['tp_early_cure']
    tp['early_cure'] = max(0.0, min(1.0, r_early_cure))
    tp['early_stay'] = max(0.0, 1.0 - tp['early_cure'])

    # ---- Local (Surg→Chemo) → DFS_L / stay ----
    # 図のとおり Local→DFS_L=1.0、Local→Metastaticなし
    tp['local_dfs_l'] = max(0.0, min(1.0, p['tp_local_dfs_l']))
    tp['local_stay']  = max(0.0, 1.0 - tp['local_dfs_l'])

    # ---- DFS_L → Healthy / Metastatic / stay ----
    r_dfsl_healthy = survival_to_annual(p['tp_dfsl_healthy_t'], p['tp_dfsl_healthy_s'])
    r_dfsl_meta    = survival_to_annual(p['tp_dfsl_meta_t'],    p['tp_dfsl_meta_s'])
    r_dfsl_stay    = max(0.0, 1.0 - r_dfsl_healthy - r_dfsl_meta)
    dfsl = _normalize({'healthy': r_dfsl_healthy, 'meta': r_dfsl_meta, 'stay': r_dfsl_stay})
    tp['dfsl_healthy'] = dfsl['healthy']
    tp['dfsl_meta']    = dfsl['meta']
    tp['dfsl_stay']    = dfsl['stay']

    # ---- Metastatic (Chemo) → DFS_M / BSC / Healthy / stay ----
    r_meta_dfsm    = survival_to_annual(p['tp_meta_dfsm_t'],    p['tp_meta_dfsm_s'])
    r_meta_bsc     = survival_to_annual(p['tp_meta_bsc_t'],     p['tp_meta_bsc_s'])
    r_meta_healthy = survival_to_annual(p['tp_meta_healthy_t'], p['tp_meta_healthy_s'])
    r_meta_stay    = max(0.0, 1.0 - r_meta_dfsm - r_meta_bsc - r_meta_healthy)
    meta = _normalize({
        'dfsm': r_meta_dfsm, 'bsc': r_meta_bsc,
        'healthy': r_meta_healthy, 'stay': r_meta_stay
    })
    tp['meta_dfsm']    = meta['dfsm']
    tp['meta_bsc']     = meta['bsc']
    tp['meta_healthy'] = meta['healthy']
    tp['meta_stay']    = meta['stay']

    # ---- DFS_M → Healthy / Metastatic / stay ----
    r_dfsm_healthy = survival_to_annual(p['tp_dfsm_healthy_t'], p['tp_dfsm_healthy_s'])
    r_dfsm_meta    = survival_to_annual(p['tp_dfsm_meta_t'],    p['tp_dfsm_meta_s'])
    r_dfsm_stay    = max(0.0, 1.0 - r_dfsm_healthy - r_dfsm_meta)
    dfsm = _normalize({'healthy': r_dfsm_healthy, 'meta': r_dfsm_meta, 'stay': r_dfsm_stay})
    tp['dfsm_healthy'] = dfsm['healthy']
    tp['dfsm_meta']    = dfsm['meta']
    tp['dfsm_stay']    = dfsm['stay']

    # ---- BSC → Death / stay ----
    r_bsc_death = survival_to_annual(p['tp_bsc_death_t'], p['tp_bsc_death_s'])
    r_bsc_stay  = max(0.0, 1.0 - r_bsc_death)
    bsc = _normalize({'death': r_bsc_death, 'stay': r_bsc_stay})
    tp['bsc_death'] = bsc['death']
    tp['bsc_stay']  = bsc['stay']

    return tp


def run_markov_model(start_age, end_age, discount_rate, tp, utilities, costs_annual,
                     initial_state='healthy', has_surveillance=True, return_trace=False,
                     costs_on_entry=None):
    """
    Markovコホートモデルを実行（新Markovモデル対応）

    States: healthy, adenoma, early, local, dfs_l, metastatic, dfs_m, bsc, death

    【設計方針】
    - cycle_length = 1年 を明示し、QALY・費用に乗算する。
    - 半サイクル補正（half-cycle correction）を適用する。
    - costs_annual: 状態にいる限り毎サイクル発生する年間費用。
    - costs_on_entry: 状態に新規進入したサイクルのみ発生する一回限りの費用。
    - 背景死亡を competing risk として Step2 で適用。
    """
    CYCLE_LENGTH = 1.0
    if costs_on_entry is None:
        costs_on_entry = {}

    states = ['healthy', 'adenoma', 'early', 'local', 'dfs_l', 'metastatic', 'dfs_m', 'bsc', 'death']
    n_states = len(states)
    idx = {s: i for i, s in enumerate(states)}

    cohort = np.zeros(n_states)
    cohort[idx[initial_state]] = 1.0

    total_cost = 0.0
    total_qaly = 0.0
    trace_history = []

    ages = list(range(start_age, end_age + 1))
    n_cycles = len(ages)

    # ── 初期状態への進入時費用をサイクル0で適用 ────────────────
    c_entry_initial = costs_on_entry.get(initial_state, 0.0)
    if c_entry_initial > 0:
        hc_weight_init = 0.5 if n_cycles > 1 else 1.0
        total_cost += c_entry_initial * 1.0 * hc_weight_init

    for i_cycle, age in enumerate(ages):
        discount_factor = 1.0 / ((1 + discount_rate) ** i_cycle)
        q_bg         = get_background_mortality(age)
        p_survive_bg = 1.0 - q_bg
        norm_util    = get_utility_norm(age)

        if return_trace:
            trace_history.append({'age': age, **{s: cohort[idx[s]] for s in states}})

        # ── Step1: Disease-specific transitions ─────────────────────
        new_cohort  = np.zeros(n_states)
        new_entries = np.zeros(n_states)

        # Healthy → Adenoma / Early / Local / Metastatic / stay
        h = cohort[idx['healthy']]
        new_cohort[idx['adenoma']]    += h * tp['healthy_to_adn']
        new_entries[idx['adenoma']]   += h * tp['healthy_to_adn']
        new_cohort[idx['early']]      += h * tp['healthy_to_early']
        new_entries[idx['early']]     += h * tp['healthy_to_early']
        new_cohort[idx['local']]      += h * tp['healthy_to_local']
        new_entries[idx['local']]     += h * tp['healthy_to_local']
        new_cohort[idx['metastatic']] += h * tp['healthy_to_meta']
        new_entries[idx['metastatic']] += h * tp['healthy_to_meta']
        new_cohort[idx['healthy']]    += h * tp['healthy_stay']

        # Adenoma → Healthy / stay  (図: Adenoma→Healthy=1.0)
        a = cohort[idx['adenoma']]
        new_cohort[idx['healthy']] += a * tp['adn_healthy']
        new_entries[idx['healthy']] += a * tp['adn_healthy']
        new_cohort[idx['adenoma']] += a * tp['adn_stay']

        # Early (Endoscopy) → Healthy(治癒) / stay  (図: Early→Healthy=1.0)
        e = cohort[idx['early']]
        new_cohort[idx['healthy']] += e * tp['early_cure']
        new_entries[idx['healthy']] += e * tp['early_cure']
        new_cohort[idx['early']]   += e * tp['early_stay']

        # Local (Surg→Chemo) → DFS_L / stay  (図: Local→DFS_L=1.0)
        loc = cohort[idx['local']]
        new_cohort[idx['dfs_l']]  += loc * tp['local_dfs_l']
        new_entries[idx['dfs_l']] += loc * tp['local_dfs_l']
        new_cohort[idx['local']]  += loc * tp['local_stay']

        # DFS_L → Healthy / Metastatic / stay
        dl = cohort[idx['dfs_l']]
        new_cohort[idx['healthy']]    += dl * tp['dfsl_healthy']
        new_entries[idx['healthy']]   += dl * tp['dfsl_healthy']
        new_cohort[idx['metastatic']] += dl * tp['dfsl_meta']
        new_entries[idx['metastatic']] += dl * tp['dfsl_meta']
        new_cohort[idx['dfs_l']]      += dl * tp['dfsl_stay']

        # Metastatic (Chemo) → DFS_M / BSC / Healthy / stay
        meta = cohort[idx['metastatic']]
        new_cohort[idx['dfs_m']]      += meta * tp['meta_dfsm']
        new_entries[idx['dfs_m']]     += meta * tp['meta_dfsm']
        new_cohort[idx['bsc']]        += meta * tp['meta_bsc']
        new_entries[idx['bsc']]       += meta * tp['meta_bsc']
        new_cohort[idx['healthy']]    += meta * tp['meta_healthy']
        new_entries[idx['healthy']]   += meta * tp['meta_healthy']
        new_cohort[idx['metastatic']] += meta * tp['meta_stay']

        # DFS_M → Healthy / Metastatic / stay
        dm = cohort[idx['dfs_m']]
        new_cohort[idx['healthy']]    += dm * tp['dfsm_healthy']
        new_entries[idx['healthy']]   += dm * tp['dfsm_healthy']
        new_cohort[idx['metastatic']] += dm * tp['dfsm_meta']
        new_entries[idx['metastatic']] += dm * tp['dfsm_meta']
        new_cohort[idx['dfs_m']]      += dm * tp['dfsm_stay']

        # BSC → Death / stay
        b = cohort[idx['bsc']]
        new_cohort[idx['death']] += b * tp['bsc_death']
        new_cohort[idx['bsc']]   += b * tp['bsc_stay']

        # Death は吸収状態
        new_cohort[idx['death']] += cohort[idx['death']]

        # ── Step2: Background mortality を competing risk として適用 ──
        final_cohort = np.zeros(n_states)
        for s, i in idx.items():
            if s == 'death':
                continue
            died_bg = new_cohort[i] * q_bg
            final_cohort[i]            += new_cohort[i] * p_survive_bg
            final_cohort[idx['death']] += died_bg
        final_cohort[idx['death']] += new_cohort[idx['death']]

        # ── 半サイクル補正 ────────────────────────────────────────
        if i_cycle == 0 or i_cycle == n_cycles - 1:
            hc_weight = 0.5
        else:
            hc_weight = 1.0

        # ── QALY ────────────────────────────────────────────
        for s, i in idx.items():
            if s == 'death':
                continue
            total_qaly += (final_cohort[i] * utilities.get(s, 0.0) * norm_util
                           * CYCLE_LENGTH * discount_factor * hc_weight)

        # ── 年間状態費用（毎サイクル）─────────────────────────
        for s, i in idx.items():
            if s == 'death':
                continue
            c = costs_annual.get(s, 0.0)
            if has_surveillance and s == 'healthy':
                c += costs_annual.get('colonoscopy', 0.0)
            total_cost += final_cohort[i] * c * CYCLE_LENGTH * discount_factor * hc_weight

        # ── 遷移時費用（新規進入時のみ・一回限り）────────────
        for s, i in idx.items():
            if s == 'death':
                continue
            c_entry = costs_on_entry.get(s, 0.0)
            if c_entry > 0 and new_entries[i] > 0:
                total_cost += new_entries[i] * c_entry * discount_factor * hc_weight

        cohort = final_cohort

    if return_trace:
        return total_cost, total_qaly, pd.DataFrame(trace_history)
    return total_cost, total_qaly

def compute_diagnostic_pathway(params, is_proband=True):
    """
    決定木に基づく診断経路コストとLynch同定率を計算

    【修正】IHC特異度・BRAF特異度を実装
      - IHC偽陽性 (pMMR患者がIHC+): (1-ihc_spec) × (1-mmr_deficient_rate)
        → 不必要にBRAF検査・遺伝カウンセリングへ進み費用が増加
      - BRAF偽陽性 (Lynch患者がBRAF V600E+と誤判定): braf_lynch_in × (1-braf_spec)
        → Lynch患者が散発性と誤分類されて見逃される

    braf_sens の意味:
      MLH1欠損の非Lynch患者(散発性)においてBRAF V600Eを正しく検出する感度。
      BRAF V600E陽性率 × テスト感度 の複合として用いる。

    Returns: (total_diag_cost, cascade_cost_for_relatives, lynch_identified_rate)
    """
    p = params
    lynch_prev    = p['lynch_prevalence']
    mmr_pos_rate  = p['mmr_deficient_rate']
    ihc_sens      = p['ihc_sens']
    ihc_spec      = p['ihc_spec']
    mlh1_neg_rate = p['mlh1_rate']
    braf_sens     = p['braf_sens']   # 非Lynch MLH1欠損例でBRAF V600Eを検出する感度
    braf_spec     = p['braf_spec']   # Lynch MLH1欠損例をBRAF陰性と正しく判定する特異度
    cascade_spec  = p.get('cascade_spec', 1.0)  # Bug 3修正: カスケード検査特異度（偽陽性補正用）

    # ── IHC ────────────────────────────────────────────────────────────
    # 真のdMMR患者を正しくIHC陽性と判定（真陽性）
    ihc_tp = ihc_sens * mmr_pos_rate
    # pMMR患者を誤ってIHC陽性と判定（偽陽性）— 特異度が<1の場合に発生
    ihc_fp = (1.0 - ihc_spec) * (1.0 - mmr_pos_rate)
    # 全IHC陽性
    ihc_pos = ihc_tp + ihc_fp

    cost_ihc_total = p['cost_ihc']

    # ── BRAF (MLH1欠損パターンのIHC陽性例のみ) ─────────────────────────
    # BRAF検査に進む患者数（MLH1欠損パターンのIHC陽性）
    braf_tested     = ihc_pos * mlh1_neg_rate
    cost_braf_total = braf_tested * p['cost_braf']

    # Lynch患者のうちMLH1欠損でIHC真陽性だった数
    ihc_tp_lynch  = ihc_sens * lynch_prev
    braf_lynch_in = ihc_tp_lynch * mlh1_neg_rate

    # 非Lynch dMMR + IHC偽陽性のMLH1欠損患者（散発性候補）
    braf_nonlynch_in = braf_tested - braf_lynch_in

    # BRAF V600E陽性と判定されて散発性として除外される非Lynch患者（真陽性）
    braf_pos_sporadic = braf_nonlynch_in * braf_sens

    # Lynch患者がBRAF偽陽性と誤判定される（見逃しに直結）
    braf_fp_lynch = braf_lynch_in * (1.0 - braf_spec)

    # 遺伝カウンセリング候補
    gc_from_braf_nonlynch = braf_nonlynch_in * (1.0 - braf_sens)  # BRAF陰性の非Lynch
    gc_from_braf_lynch    = braf_lynch_in * braf_spec               # BRAF正しく陰性のLynch
    gc_non_mlh1           = ihc_pos * (1.0 - mlh1_neg_rate)         # 非MLH1欠損のIHC陽性
    gc_candidates = gc_from_braf_nonlynch + gc_from_braf_lynch + gc_non_mlh1

    if is_proband:
        gc_uptake = p['gc_uptake_proband']
        gt_uptake = p['gt_uptake_proband']
    else:
        gc_uptake = p['gc_uptake_relative']
        gt_uptake = p['gt_uptake_relative']

    gc_attended   = gc_candidates * gc_uptake
    cost_gc_total = gc_attended * p['cost_gc']

    gt_done       = gc_attended * gt_uptake
    cost_gt_total = gt_done * (p['cost_germline'] if is_proband else p['cost_cascade'])

    if not is_proband:
        # Bug 3修正: cascade_spec を適用して偽陽性（非保因者をLynchと誤同定）を補正
        lynch_identified = float(np.clip(
            gc_uptake * gt_uptake * p['cascade_sens'] * cascade_spec * p['relative_carrier_rate'],
            0.0, 1.0
        ))
    else:
        # GCに進んだ中のLynch割合（BRAF偽陽性で見逃された分を除く）
        lynch_in_gc = gc_from_braf_lynch + ihc_tp_lynch * (1.0 - mlh1_neg_rate)
        lynch_fraction_in_gc = float(np.clip(
            lynch_in_gc / max(gc_candidates, 1e-10), 0.0, 1.0))
        lynch_identified_abs = gt_done * p['germline_sens'] * lynch_fraction_in_gc
        lynch_identified = float(np.clip(
            lynch_identified_abs / max(lynch_prev, 1e-10), 0.0, 1.0))

    cascade_cost_relatives = (
        p['num_relatives']
        * p['gc_uptake_relative']
        * p['gt_uptake_relative']
        * p['cost_cascade']
    )

    if is_proband:
        total_diag_cost = cost_ihc_total + cost_braf_total + cost_gc_total + cost_gt_total
    else:
        total_diag_cost = (
            p['gc_uptake_relative'] * p['cost_gc']
            + p['gc_uptake_relative'] * p['gt_uptake_relative'] * p['cost_cascade']
        )

    return total_diag_cost, cascade_cost_relatives, lynch_identified

def run_cea(params, is_proband=True):
    """
    費用対効果分析を実行
    Strategy A: 診断あり + サーベイランスあり
    Strategy B: 診断なし + サーベイランスなし (比較対照)
    """
    p = params
    start_age = p['proband_age'] if is_proband else p['relative_age']
    end_age = p['surveillance_end_age']
    dr = p['discount_rate']

    tp_surv   = compute_transition_probs(p, has_surveillance=True)
    tp_nosurv = compute_transition_probs(p, has_surveillance=False)

    utilities = {
        'healthy':    p['util_healthy'],
        'adenoma':    p['util_adenoma'],
        'early':      p['util_early'],
        'local':      p['util_local'],
        'dfs_l':      p['util_dfs_l'],
        'metastatic': p['util_metastatic'],
        'dfs_m':      p['util_dfs_m'],
        'bsc':        p['util_bsc'],
        'death':      0.0,
    }

    costs_annual_surv = {
        'healthy':    0,
        'adenoma':    p['cost_colonoscopy'],
        'early':      0,        # 内視鏡治療は costs_on_entry で一回のみ
        'local':      0,        # 手術は costs_on_entry で一回のみ
        'dfs_l':      0,
        'metastatic': p['cost_chemo'],   # 化学療法は継続的な年間費用
        'dfs_m':      0,
        'bsc':        p['cost_bsc'],
        'colonoscopy': p['cost_colonoscopy'],
    }
    costs_entry_surv = {
        'early': p['cost_endoscopic_tx'],   # 内視鏡治療: 一回限り
        'local': p['cost_surgery'],         # 手術: 一回限り
    }
    costs_annual_no_surv = {
        'healthy':    0,
        'adenoma':    0,
        'early':      0,
        'local':      0,
        'dfs_l':      0,
        'metastatic': p['cost_chemo'],
        'dfs_m':      0,
        'bsc':        p['cost_bsc'],
        'colonoscopy': 0,
    }
    costs_entry_no_surv = {
        'early': p['cost_endoscopic_tx'],
        'local': p['cost_surgery'],
    }

    ns_healthy    = p['no_surv_healthy']
    ns_adenoma    = p['no_surv_adenoma']
    ns_early      = p['no_surv_early']
    ns_local      = p['no_surv_local']
    ns_metastatic = p['no_surv_metastatic']
    total_ns = ns_healthy + ns_adenoma + ns_early + ns_local + ns_metastatic
    if total_ns > 0:
        ns_healthy    /= total_ns
        ns_adenoma    /= total_ns
        ns_early      /= total_ns
        ns_local      /= total_ns
        ns_metastatic /= total_ns

    # 血縁者の戦略Bに保因率を考慮
    if not is_proband:
        carrier_rate = float(np.clip(p.get('relative_carrier_rate', 1.0), 0.0, 1.0))
        non_carrier  = 1.0 - carrier_rate
        _ns_states = [
            ('healthy',    ns_healthy    * carrier_rate + non_carrier),
            ('adenoma',    ns_adenoma    * carrier_rate),
            ('early',      ns_early      * carrier_rate),
            ('local',      ns_local      * carrier_rate),
            ('metastatic', ns_metastatic * carrier_rate),
        ]
    else:
        _ns_states = [
            ('healthy',    ns_healthy),
            ('adenoma',    ns_adenoma),
            ('early',      ns_early),
            ('local',      ns_local),
            ('metastatic', ns_metastatic),
        ]

    # Strategy A
    diag_cost, cascade_cost_rel, lynch_rate = compute_diagnostic_pathway(p, is_proband)

    identified_fraction = float(np.clip(lynch_rate, 0.0, 1.0))

    if not is_proband:
        surv_uptake = float(np.clip(p.get('surveillance_uptake_relative', 1.0), 0.0, 1.0))
        identified_fraction_surv    = identified_fraction * surv_uptake
        identified_fraction_no_surv = identified_fraction * (1.0 - surv_uptake)
    else:
        identified_fraction_surv    = identified_fraction
        identified_fraction_no_surv = 0.0

    missed_fraction = 1.0 - identified_fraction

    # ① 同定されたLynch患者 → サーベイランスあり・healthy スタート
    surv_cost_identified, qaly_identified = run_markov_model(
        start_age, end_age, dr, tp_surv, utilities,
        costs_annual_surv, 'healthy', True,
        costs_on_entry=costs_entry_surv
    )

    # ② サーベイランスなし（見逃し分 & 戦略B）の Markov 計算
    no_surv_results = {}
    for init_state, weight in _ns_states:
        if weight > 0 and init_state not in no_surv_results:
            c, q = run_markov_model(start_age, end_age, dr, tp_nosurv, utilities,
                                    costs_annual_no_surv, init_state, False,
                                    costs_on_entry=costs_entry_no_surv)
            no_surv_results[init_state] = (c, q)

    cost_missed = sum(no_surv_results[s][0] * w
                      for s, w in _ns_states
                      if w > 0 and s in no_surv_results)
    qaly_missed = sum(no_surv_results[s][1] * w
                      for s, w in _ns_states
                      if w > 0 and s in no_surv_results)

    surv_cost_a = (
        identified_fraction_surv    * surv_cost_identified
        + (identified_fraction_no_surv + missed_fraction) * cost_missed
    )
    qaly_a = (
        identified_fraction_surv    * qaly_identified
        + (identified_fraction_no_surv + missed_fraction) * qaly_missed
    )
    total_cost_a = diag_cost + surv_cost_a
    if is_proband:
        total_cost_a += cascade_cost_rel

    # Strategy B
    total_cost_b = sum(no_surv_results[s][0] * w
                       for s, w in _ns_states
                       if w > 0 and s in no_surv_results)
    total_qaly_b = sum(no_surv_results[s][1] * w
                       for s, w in _ns_states
                       if w > 0 and s in no_surv_results)

    delta_cost = total_cost_a - total_cost_b
    delta_qaly = qaly_a - total_qaly_b
    icer = delta_cost / delta_qaly if delta_qaly != 0 else float('inf')

    return {
        'cost_a': total_cost_a,
        'qaly_a': qaly_a,
        'cost_b': total_cost_b,
        'qaly_b': total_qaly_b,
        'delta_cost': delta_cost,
        'delta_qaly': delta_qaly,
        'icer': icer,
        'diag_cost': diag_cost,
        'cascade_cost_relatives': cascade_cost_rel,
        'lynch_rate': lynch_rate,
        'surv_cost_a': surv_cost_a,
        'identified_fraction': identified_fraction,
        'identified_fraction_surv': identified_fraction_surv,
        'identified_fraction_no_surv': identified_fraction_no_surv,
        'missed_fraction': missed_fraction,
    }

# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def _pert_sample(lo, hi, mode, rng):
    """β-PERT(lo, hi, mode) 分布からサンプリング。"""
    mean = (lo + 4.0 * mode + hi) / 6.0
    span = hi - lo
    if span < 1e-12:
        return float(mean)
    a = 6.0 * (mean - lo) / span
    b = 6.0 * (hi - mean) / span
    a = max(a, 0.5)
    b = max(b, 0.5)
    return float(np.clip(lo + rng.beta(a, b) * span, lo, hi))


def run_psa(params, is_proband=True, n_runs=1000, seed=42, wtp=5_000_000):
    """確率的感度分析 (PSA) + EVPI計算用NMBを返す"""
    rng = np.random.default_rng(seed)
    results = []

    for _ in range(n_runs):
        p = deepcopy(params)
        p['ihc_sens']      = _pert_sample(params.get('sa_ihc_sens_lo', 0.85),
                                          params.get('sa_ihc_sens_hi', 0.95),
                                          params['ihc_sens'], rng)
        p['ihc_spec']      = _pert_sample(params.get('sa_ihc_spec_lo', 0.75),
                                          params.get('sa_ihc_spec_hi', 0.90),
                                          params['ihc_spec'], rng)
        p['braf_sens']     = _pert_sample(params.get('sa_braf_sens_lo', 0.60),
                                          params.get('sa_braf_sens_hi', 0.80),
                                          params['braf_sens'], rng)
        p['braf_spec']     = _pert_sample(params.get('sa_braf_spec_lo', 0.90),
                                          params.get('sa_braf_spec_hi', 0.98),
                                          params['braf_spec'], rng)
        p['germline_sens'] = 1.00
        p['germline_spec'] = 1.00
        p['discount_rate'] = float(np.clip(rng.normal(0.02, 0.005), 0.005, 0.10))
        # ユーティリティ（dfs_l と dfs_m は独立してサンプリング）
        p['util_healthy']    = float(np.clip(rng.beta(99, 1),   0.50, 1.00))
        p['util_adenoma']    = float(np.clip(rng.beta(98, 2),   0.50, 1.00))
        p['util_early']      = 0.95
        p['util_local']      = _pert_sample(0.65, 0.75, 0.70, rng)
        p['util_metastatic'] = _pert_sample(0.55, 0.65, 0.60, rng)
        p['util_dfs_l']      = _pert_sample(0.70, 0.80, 0.75, rng)  # 修正: dfs_l も独立してサンプリング
        p['util_dfs_m']      = _pert_sample(0.70, 0.80, 0.75, rng)
        p['util_bsc']        = _pert_sample(0.20, 0.30, 0.25, rng)
        # 血縁者サーベイランス参加率
        p['surveillance_uptake_relative'] = _pert_sample(
            params.get('sa_surv_uptake_rel_lo', 0.67),
            params.get('sa_surv_uptake_rel_hi', 0.87),
            params['surveillance_uptake_relative'], rng)
        # Markov 遷移確率の不確実性
        p['tp_healthy_adn_surv_s']   = float(np.clip(rng.beta(6, 4),  0.01, 0.99))
        p['tp_healthy_adn_nosurv_s'] = float(np.clip(rng.beta(7, 3),  0.01, 0.99))
        p['tp_adenoma_healthy']      = float(np.clip(rng.beta(9, 1),  0.50, 0.99))
        p['tp_early_cure']           = float(np.clip(rng.beta(49, 1), 0.90, 1.00))
        # 注: tp_adn_early_s, tp_early_local_s は compute_transition_probs で未使用のため削除

        res = run_cea(p, is_proband)
        nmb = wtp * res['delta_qaly'] - res['delta_cost']
        results.append({
            'delta_cost':  res['delta_cost'],
            'delta_qaly':  res['delta_qaly'],
            'icer':        res['icer'],
            'nmb':         nmb,
            'nmb_a': wtp * res['qaly_a'] - res['cost_a'],
            'nmb_b': wtp * res['qaly_b'] - res['cost_b'],
        })

    return pd.DataFrame(results)

# ============================================================
