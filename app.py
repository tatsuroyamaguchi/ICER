"""
Lynch症候群の診断と大腸癌サーベイランスに関する費用対効果分析
Streamlit App  ―  GitHub Pages / stlite 対応版
"""

# 1. Standard Library
from copy import deepcopy

# 2. Third-party Libraries
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
try:
    import japanize_matplotlib          # ローカル環境では日本語フォントを有効化
except ImportError:
    # stlite (Pyodide) 環境では japanize_matplotlib が非対応のため、
    # matplotlib の rcParams でフォールバックフォントを設定する
    matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False

import numpy as np
import pandas as pd
import streamlit as st

# 3. Local Modules — 計算エンジン
from cea_core import (
    MORTALITY_TABLE_2024,
    get_background_mortality,
    get_utility_norm,
    survival_to_annual,
    _normalize,
    compute_transition_probs,
    run_markov_model,
    compute_diagnostic_pathway,
    run_cea,
    _pert_sample,
    run_psa,
)

# ============================================================
# PAGE CONFIG — 必ず最初の st コールにする
# ============================================================
st.set_page_config(
    page_title="Lynch症候群 CEA: IHC/BRAF Flow",
    page_icon="🧬",
    layout="wide",
)

st.title("💰️ Cost-Effectiveness Analysis")
st.subheader(" Diagnosis of Lynch Syndrome and CRC Surveillance")

# ============================================================
# SIDEBAR PARAMETERS
# ============================================================
st.sidebar.header("⚙️ パラメーター設定")

# ────────────────────────────────────────────────────────────
# A. 共通設定
# ────────────────────────────────────────────────────────────
with st.sidebar.expander("📊 共通設定", expanded=False):
    discount_rate = st.number_input("割引率 (年率)", value=0.02, min_value=0.0, max_value=0.10, step=0.005, format="%.3f")
    proband_age = st.number_input("発端者 開始年齢", value=50, min_value=18, max_value=80)
    relative_age = st.number_input("血縁者 開始年齢", value=25, min_value=18, max_value=80)
    surveillance_end_age = st.number_input("サーベイランス終了年齢", value=85, min_value=50, max_value=100)
    simulation_runs = st.number_input("モンテカルロ試行回数", value=1000, min_value=100, max_value=50000, step=100)
    random_seed = st.number_input("乱数シード", value=42, min_value=0, max_value=9999)

# ────────────────────────────────────────────────────────────
# B. 決定木 (GT.jpg) 関連パラメーター
# ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🌳 決定木パラメーター")

with st.sidebar.expander("🧪 診断精度（決定木）", expanded=False):
    st.caption("決定木の各検査ノードに対応する感度・特異度")
    st.markdown("**感度　　　特異度**")

    col1, col2 = st.columns(2)
    with col1:
        st.caption("感度")
    with col2:
        st.caption("特異度")

    col1, col2 = st.columns(2)
    st.markdown("🔬 **IHC**")
    col1, col2 = st.columns(2)
    with col1:
        ihc_sens = st.number_input("感度##ihc", value=0.91, min_value=0.50, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")
    with col2:
        ihc_spec = st.number_input("特異度##ihc", value=0.83, min_value=0.50, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")

    st.markdown("🧫 **BRAF**")
    col1, col2 = st.columns(2)
    with col1:
        braf_sens = st.number_input("感度##braf", value=0.76, min_value=0.50, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")
    with col2:
        braf_spec = st.number_input("特異度##braf", value=0.96, min_value=0.50, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")

    st.markdown("🧬 **生殖細胞系列検査**")
    col1, col2 = st.columns(2)
    with col1:
        germline_sens = st.number_input("感度##germ", value=1.00, min_value=0.90, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")
    with col2:
        germline_spec = st.number_input("特異度##germ", value=1.00, min_value=0.90, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")

    st.markdown("🔗 **カスケード検査**")
    col1, col2 = st.columns(2)
    with col1:
        cascade_sens = st.number_input("感度##cas", value=1.00, min_value=0.90, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")
    with col2:
        cascade_spec = st.number_input("特異度##cas", value=1.00, min_value=0.90, max_value=1.00, step=0.01, format="%.2f", label_visibility="collapsed")

with st.sidebar.expander("🧬 有病率・遺伝子割合（決定木）", expanded=False):
    st.caption("決定木の確率ノードに対応する有病率・割合")
    lynch_prevalence = st.number_input("Lynch症候群 有病率 (大腸癌中)", value=0.01, min_value=0.001, max_value=0.10, step=0.001, format="%.3f")
    mmr_deficient_rate = st.number_input("IHC陽性割合 (dMMR率)", value=0.0494, min_value=0.001, max_value=0.20, step=0.001, format="%.4f")
    st.markdown("**MMR遺伝子別割合** *(合計=1.0)*")
    col1, col2 = st.columns(2)
    with col1:
        mlh1_rate = st.number_input("MLH1", value=0.30, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        msh6_rate = st.number_input("MSH6", value=0.30, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    with col2:
        msh2_rate = st.number_input("MSH2", value=0.30, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
        pms2_rate = st.number_input("PMS2", value=0.10, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    _gene_total = mlh1_rate + msh2_rate + msh6_rate + pms2_rate
    if abs(_gene_total - 1.0) > 0.01:
        st.warning(f"⚠️ 合計 = {_gene_total:.2f}（1.0 になるよう調整してください）")

with st.sidebar.expander("👥 受検率・血縁者（決定木）", expanded=False):
    st.caption("決定木の「Decline / 受検」分岐に対応する受検率")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**発端者**")
        gc_uptake_proband = st.number_input("遺伝カウンセリング", value=0.84, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="gc_prob")
        gt_uptake_proband = st.number_input("遺伝子検査", value=0.80, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="gt_prob")
    with col2:
        st.markdown("**血縁者**")
        gc_uptake_relative = st.number_input("遺伝カウンセリング", value=0.95, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="gc_rel")
        gt_uptake_relative = st.number_input("遺伝子検査", value=0.60, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="gt_rel")
    num_relatives = st.number_input("第１度近親者数", value=3, min_value=1, max_value=10)
    relative_carrier_rate = st.number_input("血縁者の保因率", value=0.50, min_value=0.0, max_value=1.0, step=0.05, format="%.2f")
    surveillance_uptake_relative = st.number_input(
        "サーベイランス参加率 (血縁者)",
        value=0.79, min_value=0.0, max_value=1.0, step=0.01, format="%.2f",
        help="Lynch症候群と診断された血縁者のうち実際にサーベイランス（大腸内視鏡）を受ける割合。\n"
             "中央値 0.79（95%CI: 0.67–0.87）"
    )

with st.sidebar.expander("💰 費用 — 決定木・診断（円）", expanded=False):
    st.caption("決定木ノードに対応する検査・診断費用")
    cost_ihc = st.number_input("IHC検査", value=27000, min_value=0, step=1000)
    cost_braf = st.number_input("BRAF検査", value=25000, min_value=0, step=1000)
    cost_germline = st.number_input("生殖細胞系列検査", value=202000, min_value=0, step=1000)
    cost_cascade = st.number_input("カスケード検査", value=38800, min_value=0, step=1000)
    cost_gc = st.number_input("遺伝カウンセリング", value=10000, min_value=0, step=1000)

# ────────────────────────────────────────────────────────────
# C. Markovモデル (Markov.jpg) 関連パラメーター
# ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🔄 Markovモデルパラメーター")

with st.sidebar.expander("🏥 Utility・効用値（Markov）", expanded=False):
    st.caption("各Markov状態ノードに記載の効用値（赤字）")
    st.info(
        "💡 **効用値の設計について**\n\n"
        "ここで設定する値は年齢別国民標準値（EQ-5D norm）への **乗数** として機能します。\n"
        "例: Healthy=1.0 → norm値そのまま、Middle=0.69 → norm値×0.69\n\n"
        "文献の絶対EQ-5D値を直接入力する場合は、\n"
        "国民標準値に対する**相対比**（文献値 ÷ 年齢相当norm値）を設定してください。"
    )
    col1, col2 = st.columns(2)
    with col1:
        util_healthy     = st.number_input("Healthy",           value=1.0,  min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_early       = st.number_input("Early (Endoscopy)", value=0.95, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_dfs_l       = st.number_input("DFS_L",             value=0.75, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_dfs_m       = st.number_input("DFS_M",             value=0.75, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_death       = 0.0
        st.caption("Death: 効用値 = 0.0（固定）")
    with col2:
        util_adenoma     = st.number_input("Adenoma",           value=0.99, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_local       = st.number_input("Local (手術→化療)",value=0.70,  min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_metastatic  = st.number_input("Metastatic (化療)", value=0.60, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        util_bsc         = st.number_input("BSC",               value=0.25, min_value=0.0, max_value=1.0, step=0.01, format="%.2f")

with st.sidebar.expander("💰 費用 — Markov治療（円）", expanded=False):
    st.caption("Markov各状態で発生する治療・管理費用")
    cost_colonoscopy    = st.number_input("大腸内視鏡検査 (年1回)",        value=15500,   min_value=0, step=500)
    cost_endoscopic_tx  = st.number_input("内視鏡治療 (Early/Endoscopy)",  value=150000,  min_value=0, step=10000)
    cost_surgery        = st.number_input("手術 (Local/Surg→Chemo)",       value=750000,  min_value=0, step=10000)
    cost_chemo          = st.number_input("化学療法 (Metastatic)",          value=2000000, min_value=0, step=100000)
    cost_bsc            = st.number_input("BSC (年間)",                    value=200000,  min_value=0, step=10000)

with st.sidebar.expander("📈 遷移確率 — Markov（X[t, S]）", expanded=False):
    st.caption(
        "各矢印の遷移確率は X[t, S] = 1-(1-S)^(1/t) で年次確率に変換されます。\n"
        "t=観察年数、S=t年累積遷移率"
    )

    _h1, _h2, _h3 = st.columns([3, 1, 1])
    with _h1: st.markdown("**遷移**")
    with _h2: st.markdown("**t 年**")
    with _h3: st.markdown("**S 累積率**")

    st.markdown("---")
    st.markdown("**🩺 サーベイランスあり（内視鏡）**")
    st.caption("内視鏡サーベイランスにより早期発見される場合の遷移確率")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Adenoma (サーベイランスあり)")
    with _c2: tp_healthy_adn_surv_t = st.number_input("t##hadn_s_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="hadn_s_t")
    with _c3: tp_healthy_adn_surv_s = st.number_input("S##hadn_s_s", value=0.6, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="hadn_s_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Early ※サーベイランスあり")
    with _c2: tp_healthy_early_t = st.number_input("t##he_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="he_t")
    with _c3: tp_healthy_early_s = st.number_input("S##he_s", value=0.2, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="he_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Local ※サーベイランスあり")
    with _c2: tp_healthy_local_t = st.number_input("t##hl_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="hl_t")
    with _c3: tp_healthy_local_s = st.number_input("S##hl_s", value=0.08, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="hl_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Metastatic ※サーベイランスあり")
    with _c2: tp_healthy_meta_t = st.number_input("t##hm_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="hm_t")
    with _c3: tp_healthy_meta_s = st.number_input("S##hm_s", value=0.01, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="hm_s")

    st.markdown("---")
    st.markdown("**🚫 サーベイランスなし（進行癌発見↑）**")
    st.caption("症状が出るまで発見されないため Early↓、Local↑、Meta↑")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Early ※サーベイランスなし")
    with _c2: tp_healthy_early_nosurv_t = st.number_input("t##he_ns_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="he_ns_t")
    with _c3: tp_healthy_early_nosurv_s = st.number_input("S##he_ns_s", value=0.10, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="he_ns_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Local ※サーベイランスなし")
    with _c2: tp_healthy_local_nosurv_t = st.number_input("t##hl_ns_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="hl_ns_t")
    with _c3: tp_healthy_local_nosurv_s = st.number_input("S##hl_ns_s", value=0.15, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="hl_ns_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → Metastatic ※サーベイランスなし")
    with _c2: tp_healthy_meta_nosurv_t = st.number_input("t##hm_ns_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="hm_ns_t")
    with _c3: tp_healthy_meta_nosurv_s = st.number_input("S##hm_ns_s", value=0.05, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="hm_ns_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("腺腫 → 健常（直接確率、図通り=1.0）")
    with _c2: st.markdown("―")
    with _c3: tp_adenoma_healthy = st.number_input("P##adn_h", value=1.0, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="adn_h")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("Early → 健常（治癒・直接確率）")
    with _c2: st.markdown("―")
    with _c3: tp_early_cure = st.number_input("S##early_cure", value=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="early_cure")

    st.markdown("---")
    st.markdown("**🔪 Local (手術→化療) → DFS_L**")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("Local → DFS_L（直接確率、図通り=1.0）")
    with _c2: st.markdown("―")
    with _c3: tp_local_dfs_l = st.number_input("P##ld", value=1.0, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="ld")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("DFS_L → 健常")
    with _c2: tp_dfsl_healthy_t = st.number_input("t##dlh_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="dlh_t")
    with _c3: tp_dfsl_healthy_s = st.number_input("S##dlh_s", value=0.7, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="dlh_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("DFS_L → Metastatic")
    with _c2: tp_dfsl_meta_t = st.number_input("t##dlm_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="dlm_t")
    with _c3: tp_dfsl_meta_s = st.number_input("S##dlm_s", value=0.3, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="dlm_s")

    st.markdown("---")
    st.markdown("**💊 Metastatic（化学療法）**")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("Metastatic → DFS_M")
    with _c2: tp_meta_dfsm_t = st.number_input("t##md_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="md_t")
    with _c3: tp_meta_dfsm_s = st.number_input("S##md_s", value=0.4, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="md_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("Metastatic → BSC")
    with _c2: tp_meta_bsc_t = st.number_input("t##mb_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="mb_t")
    with _c3: tp_meta_bsc_s = st.number_input("S##mb_s", value=0.6, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="mb_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("Metastatic → 健常")
    with _c2: tp_meta_healthy_t = st.number_input("t##mh_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="mh_t")
    with _c3: tp_meta_healthy_s = st.number_input("S##mh_s", value=0.5, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="mh_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("DFS_M → 健常")
    with _c2: tp_dfsm_healthy_t = st.number_input("t##dmh_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="dmh_t")
    with _c3: tp_dfsm_healthy_s = st.number_input("S##dmh_s", value=0.5, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="dmh_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("DFS_M → Metastatic")
    with _c2: tp_dfsm_meta_t = st.number_input("t##dmm_t", value=5, min_value=1, max_value=20, label_visibility="collapsed", key="dmm_t")
    with _c3: tp_dfsm_meta_s = st.number_input("S##dmm_s", value=0.5, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="dmm_s")

    st.markdown("---")
    st.markdown("**🛏️ BSC・サーベイランスなし**")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("BSC → 死亡")
    with _c2: tp_bsc_death_t = st.number_input("t##bd_t", value=1, min_value=1, max_value=5, label_visibility="collapsed", key="bd_t")
    with _c3: tp_bsc_death_s = st.number_input("S##bd_s", value=0.98, min_value=0.0, max_value=1.0, step=0.01, format="%.2f", label_visibility="collapsed", key="bd_s")

    _c1, _c2, _c3 = st.columns([3, 1, 1])
    with _c1: st.markdown("健常 → 腺腫 (サーベイランスなし)")
    with _c2: tp_healthy_adn_nosurv_t = st.number_input("t##hadn_ns_t", value=10, min_value=1, max_value=20, label_visibility="collapsed", key="hadn_ns_t")
    with _c3: tp_healthy_adn_nosurv_s = st.number_input("S##hadn_ns_s", value=0.7, min_value=0.0, max_value=1.0, step=0.05, format="%.2f", label_visibility="collapsed", key="hadn_ns_s")

    st.markdown("**病期分布（サーベイランスなし・初期状態）**")
    st.caption("healthy を含む5状態で初期分布を指定（合計=1.0 に正規化されます）")
    _c1, _c2 = st.columns(2)
    with _c1:
        no_surv_healthy    = st.number_input("健常",            value=0.00,    min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        no_surv_adenoma    = st.number_input("腺腫",            value=0.00,    min_value=0.0, max_value=1.0, step=0.01, format="%.2f")
        no_surv_local      = st.number_input("Local (手術→化療)", value=105/113, min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
    with _c2:
        no_surv_early      = st.number_input("Early (内視鏡)",  value=6/113,   min_value=0.0, max_value=1.0, step=0.01, format="%.3f")
        no_surv_metastatic = st.number_input("Metastatic",      value=2/113,   min_value=0.0, max_value=1.0, step=0.01, format="%.3f")

# ────────────────────────────────────────────────────────────
# D. 感度分析設定（下限・中央値・上限を1行にまとめ）
# ────────────────────────────────────────────────────────────
st.sidebar.markdown("### 🌪️ 感度分析")

with st.sidebar.expander("🌪️ 感度分析設定（下限 ／ 上限）", expanded=False):
    st.caption("各パラメーターの不確実性範囲。左列＝下限、右列＝上限。")

    # ── 共通 ──
    st.markdown("**📊 共通**")
    _c1, _c2 = st.columns(2)
    with _c1: sa_discount_lo = st.number_input("割引率 Lo", value=0.01, format="%.3f", label_visibility="collapsed", key="sa_disc_lo")
    with _c2: sa_discount_hi = st.number_input("割引率 Hi", value=0.03, format="%.3f", label_visibility="collapsed", key="sa_disc_hi")
    st.caption(f"割引率　下限 {sa_discount_lo:.3f} ／ 上限 {sa_discount_hi:.3f}")

    # ── 決定木 ──
    st.markdown("**🌳 決定木（診断精度）**")

    def _sa_row(label, lo_key, hi_key, lo_val, hi_val, fmt="%.2f"):
        """下限・上限を2列で並べ、ラベル付きキャプションを表示するヘルパー"""
        _c1, _c2 = st.columns(2)
        with _c1: lo = st.number_input(f"{label} Lo", value=lo_val, format=fmt, label_visibility="collapsed", key=lo_key)
        with _c2: hi = st.number_input(f"{label} Hi", value=hi_val, format=fmt, label_visibility="collapsed", key=hi_key)
        st.caption(f"{label}　下限 {lo:{fmt[1:]}} ／ 上限 {hi:{fmt[1:]}}")
        return lo, hi

    sa_ihc_sens_lo, sa_ihc_sens_hi       = _sa_row("IHC 感度",              "sa_ihc_sens_lo",       "sa_ihc_sens_hi",       0.85, 0.95)
    sa_ihc_spec_lo, sa_ihc_spec_hi       = _sa_row("IHC 特異度",            "sa_ihc_spec_lo",       "sa_ihc_spec_hi",       0.75, 0.90)
    sa_braf_sens_lo, sa_braf_sens_hi     = _sa_row("BRAF 感度",             "sa_braf_sens_lo",      "sa_braf_sens_hi",      0.60, 0.80)
    sa_braf_spec_lo, sa_braf_spec_hi     = _sa_row("BRAF 特異度",           "sa_braf_spec_lo",      "sa_braf_spec_hi",      0.90, 0.98)
    sa_germline_sens_lo, sa_germline_sens_hi = _sa_row("生殖細胞系列 感度", "sa_germ_sens_lo",      "sa_germ_sens_hi",      0.99, 1.00)
    sa_germline_spec_lo, sa_germline_spec_hi = _sa_row("生殖細胞系列 特異度","sa_germ_spec_lo",     "sa_germ_spec_hi",      0.99, 1.00)
    sa_cascade_sens_lo, sa_cascade_sens_hi   = _sa_row("カスケード 感度",   "sa_cas_sens_lo",       "sa_cas_sens_hi",       0.99, 1.00)
    sa_cascade_spec_lo, sa_cascade_spec_hi   = _sa_row("カスケード 特異度", "sa_cas_spec_lo",       "sa_cas_spec_hi",       0.99, 1.00)
    sa_surv_uptake_rel_lo, sa_surv_uptake_rel_hi = _sa_row("血縁者サーベイランス参加率", "sa_surv_up_lo", "sa_surv_up_hi", 0.67, 0.87)

# ============================================================
# COLLECT ALL PARAMETERS
# ============================================================

params = {
    'discount_rate': discount_rate,

    'proband_age': proband_age,
    'relative_age': relative_age,
    'surveillance_end_age': surveillance_end_age,
    # Diagnostic accuracy
    'ihc_sens': ihc_sens, 'ihc_spec': ihc_spec,
    'braf_sens': braf_sens, 'braf_spec': braf_spec,
    'germline_sens': germline_sens, 'germline_spec': germline_spec,
    'cascade_sens': cascade_sens, 'cascade_spec': cascade_spec,
    # Prevalence
    'lynch_prevalence': lynch_prevalence,
    'mmr_deficient_rate': mmr_deficient_rate,
    'mlh1_rate': mlh1_rate,
    'msh2_rate': msh2_rate,
    'msh6_rate': msh6_rate,
    'pms2_rate': pms2_rate,
    # Uptake
    'gc_uptake_proband': gc_uptake_proband,
    'gc_uptake_relative': gc_uptake_relative,
    'gt_uptake_proband': gt_uptake_proband,
    'gt_uptake_relative': gt_uptake_relative,
    'num_relatives': num_relatives,
    'relative_carrier_rate': relative_carrier_rate,
    'surveillance_uptake_relative': surveillance_uptake_relative,
    # Costs
    'cost_ihc': cost_ihc, 'cost_braf': cost_braf,
    'cost_germline': cost_germline, 'cost_cascade': cost_cascade,
    'cost_gc': cost_gc, 'cost_colonoscopy': cost_colonoscopy,
    'cost_endoscopic_tx': cost_endoscopic_tx, 'cost_surgery': cost_surgery,
    'cost_chemo': cost_chemo, 'cost_bsc': cost_bsc,
    # Utilities
    'util_healthy': util_healthy, 'util_adenoma': util_adenoma,
    'util_early': util_early, 'util_local': util_local,
    'util_dfs_l': util_dfs_l, 'util_metastatic': util_metastatic,
    'util_dfs_m': util_dfs_m, 'util_bsc': util_bsc, 'util_death': util_death,
    # Transition probs
    'tp_adenoma_healthy': tp_adenoma_healthy,
    'tp_healthy_adn_surv_t':   tp_healthy_adn_surv_t,
    'tp_healthy_adn_surv_s':   tp_healthy_adn_surv_s,
    'tp_healthy_adn_nosurv_t': tp_healthy_adn_nosurv_t,
    'tp_healthy_adn_nosurv_s': tp_healthy_adn_nosurv_s,
    'tp_healthy_early_t': tp_healthy_early_t, 'tp_healthy_early_s': tp_healthy_early_s,
    'tp_healthy_local_t': tp_healthy_local_t, 'tp_healthy_local_s': tp_healthy_local_s,
    'tp_healthy_meta_t':  tp_healthy_meta_t,  'tp_healthy_meta_s':  tp_healthy_meta_s,
    'tp_healthy_early_nosurv_t': tp_healthy_early_nosurv_t, 'tp_healthy_early_nosurv_s': tp_healthy_early_nosurv_s,
    'tp_healthy_local_nosurv_t': tp_healthy_local_nosurv_t, 'tp_healthy_local_nosurv_s': tp_healthy_local_nosurv_s,
    'tp_healthy_meta_nosurv_t':  tp_healthy_meta_nosurv_t,  'tp_healthy_meta_nosurv_s':  tp_healthy_meta_nosurv_s,
    'tp_early_cure': tp_early_cure,
    'tp_local_dfs_l': tp_local_dfs_l,
    'tp_dfsl_healthy_t': tp_dfsl_healthy_t, 'tp_dfsl_healthy_s': tp_dfsl_healthy_s,
    'tp_dfsl_meta_t': tp_dfsl_meta_t, 'tp_dfsl_meta_s': tp_dfsl_meta_s,
    'tp_meta_dfsm_t': tp_meta_dfsm_t, 'tp_meta_dfsm_s': tp_meta_dfsm_s,
    'tp_meta_bsc_t': tp_meta_bsc_t, 'tp_meta_bsc_s': tp_meta_bsc_s,
    'tp_meta_healthy_t': tp_meta_healthy_t, 'tp_meta_healthy_s': tp_meta_healthy_s,
    'tp_dfsm_healthy_t': tp_dfsm_healthy_t, 'tp_dfsm_healthy_s': tp_dfsm_healthy_s,
    'tp_dfsm_meta_t': tp_dfsm_meta_t, 'tp_dfsm_meta_s': tp_dfsm_meta_s,
    'tp_bsc_death_t': tp_bsc_death_t, 'tp_bsc_death_s': tp_bsc_death_s,
    # No surveillance stage distribution
    'no_surv_healthy': no_surv_healthy,
    'no_surv_adenoma': no_surv_adenoma,
    'no_surv_early': no_surv_early,
    'no_surv_local': no_surv_local,
    'no_surv_metastatic': no_surv_metastatic,
    # PSA用の感度分析範囲
    'sa_ihc_sens_lo': sa_ihc_sens_lo, 'sa_ihc_sens_hi': sa_ihc_sens_hi,
    'sa_ihc_spec_lo': sa_ihc_spec_lo, 'sa_ihc_spec_hi': sa_ihc_spec_hi,
    'sa_braf_sens_lo': sa_braf_sens_lo, 'sa_braf_sens_hi': sa_braf_sens_hi,
    'sa_braf_spec_lo': sa_braf_spec_lo, 'sa_braf_spec_hi': sa_braf_spec_hi,
    'sa_germline_sens_lo': sa_germline_sens_lo, 'sa_germline_sens_hi': sa_germline_sens_hi,
    'sa_germline_spec_lo': sa_germline_spec_lo, 'sa_germline_spec_hi': sa_germline_spec_hi,
    'sa_cascade_sens_lo': sa_cascade_sens_lo, 'sa_cascade_sens_hi': sa_cascade_sens_hi,
    'sa_cascade_spec_lo': sa_cascade_spec_lo, 'sa_cascade_spec_hi': sa_cascade_spec_hi,
    'sa_surv_uptake_rel_lo': sa_surv_uptake_rel_lo,
    'sa_surv_uptake_rel_hi': sa_surv_uptake_rel_hi,
}

# ============================================================
# MAIN RESULTS
# ============================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    ["🏥 Proband", "👨‍👩‍👧 Relatives",
     "🌪️ Tornade (DSA)", "📈 PSA / CEAC / EVPI", "✅ Validation (Trace Plot)",
     "🌳 Flowchart", "📋 Parameters", "📖References"]
)

# ============================================================
# ---------- TAB1: PROBAND ----------
# ============================================================
with tab1:
    st.subheader("Cost-Effectiveness Analysis for Proband")
    st.caption("💡 すべての費用・QALYは **1人あたり（割引済み期待値）** で表示しています。")

    with st.spinner("計算中..."):
        res_p = run_cea(params, is_proband=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("戦略A 費用", f"¥{res_p['cost_a']:,.0f}")
        st.metric("QALY (A)", f"{res_p['qaly_a']:.4f}")
    with c2:
        st.metric("戦略B 費用", f"¥{res_p['cost_b']:,.0f}")
        st.metric("QALY (B)", f"{res_p['qaly_b']:.4f}")
    with c3:
        st.metric("ΔC", f"¥{res_p['delta_cost']:,.0f}")
        st.metric("ΔE (QALY)", f"{res_p['delta_qaly']:.4f}")

    icer_v = res_p['icer']
    st.metric(
        "💡 ICER ＝ ΔC / ΔE",
        f"¥{icer_v:,.0f} / QALY" if np.isfinite(icer_v) else "支配的"
    )
    st.metric("診断費用（発端者）", f"¥{res_p['diag_cost']:,.0f}")
    st.metric("Lynch同定率 (推定)", f"{res_p['lynch_rate']*100:.2f}%")
    st.metric("同定割合（サーベイランスあり）", f"{res_p['identified_fraction']*100:.1f}%")
    st.metric("見逃割合（サーベイランスなし）", f"{res_p['missed_fraction']*100:.1f}%")

    st.markdown("**費用内訳**")
    df_detail = pd.DataFrame({
        '項目': ['IHC/BRAF/GC/生殖細胞系列',
                 'サーベイランス+治療（同定分）', 'サーベイランスなし治療（見逃分）',
                 '加重平均 サーベイランス+治療費用', '合計費用'],
        '戦略A': [
            f"¥{res_p['diag_cost']:,.0f}",
            f"{res_p['identified_fraction']*100:.1f}% × (サーベイランスあり分)",
            f"{res_p['missed_fraction']*100:.1f}% × (見逃し分)",
            f"¥{res_p['surv_cost_a']:,.0f}",
            f"¥{res_p['cost_a']:,.0f}",
        ],
        '戦略B': [
            "¥0", "—", "—",
            f"¥{res_p['cost_b']:,.0f}",
            f"¥{res_p['cost_b']:,.0f}",
        ],
    })
    st.table(df_detail)

    st.subheader("年齢別EQ-5D効用値 (国民標準値 + 状態別)")
    ages = list(range(proband_age, surveillance_end_age + 1))
    norm_utils = [get_utility_norm(a) for a in ages]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(ages, norm_utils, label='National norm (EQ-5D)', color='navy')
    for state, base_u, color in [
        ('Healthy', util_healthy, 'green'), ('Adenoma', util_adenoma, 'blue'),
        ('Early (Endoscopy)', util_early, 'orange'), ('Local (Surg->Chemo)', util_local, 'red'),
        ('Metastatic (Chemo)', util_metastatic, 'darkred'), ('BSC', util_bsc, 'black'),
    ]:
        ax.plot(ages, [get_utility_norm(a) * base_u for a in ages],
                label=f'{state} (base={base_u})', linestyle='--', color=color, alpha=0.7)
    ax.set_xlabel("Age"); ax.set_ylabel("Utility")
    ax.set_title("Age-adjusted Utility by Health State")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    st.pyplot(fig); plt.close()

# ============================================================
# ---------- TAB2: RELATIVES ----------
# ============================================================
with tab2:
    st.subheader("Cost-Effectiveness Analysis for Relatives")
    with st.spinner("計算中..."):
        res_r = run_cea(deepcopy(params), is_proband=False)

    st.caption("💡 すべての費用・QALYは **1人あたり（割引済み期待値）** で表示しています。")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("戦略A: 診断+サーベイランス あり", f"¥{res_r['cost_a']:,.0f}", "1人あたり総費用 (割引済)")
        st.metric("QALY (戦略A) [1人あたり]", f"{res_r['qaly_a']:.4f}")
    with col2:
        st.metric("戦略B: サーベイランス なし", f"¥{res_r['cost_b']:,.0f}", "1人あたり総費用 (割引済)")
        st.metric("QALY (戦略B) [1人あたり]", f"{res_r['qaly_b']:.4f}")
    with col3:
        st.metric("増分費用 (ΔC) [1人あたり]", f"¥{res_r['delta_cost']:,.0f}")
        st.metric("増分QALY (ΔE) [1人あたり]", f"{res_r['delta_qaly']:.4f}")

    icer_r = res_r['icer']
    st.metric("💡 ICER (血縁者)  ＝ ΔC / ΔE",
              f"¥{icer_r:,.0f} / QALY" if np.isfinite(icer_r) else "支配的")
    st.metric("カスケード検査費用 [1人あたり]", f"¥{res_r['diag_cost']:,.0f}")
    st.metric("血縁者Lynch同定率 (推定)", f"{res_r['lynch_rate']*100:.2f}%")
    st.metric("同定後サーベイランス参加率", f"{surveillance_uptake_relative*100:.1f}%")
    st.metric("サーベイランスあり割合（同定×参加率）",
              f"{res_r['identified_fraction_surv']*100:.1f}%")
    st.metric("サーベイランスなし割合（診断後非参加 + 見逃し）",
              f"{(res_r['identified_fraction_no_surv'] + res_r['missed_fraction'])*100:.1f}%")

    st.subheader("家系全体の費用対効果試算")
    st.caption(
        "家系合計 ＝ 発端者1人 + 血縁者×人数（各値は1人あたり割引済み期待値の合算）。"
    )

    fc_a = res_p['cost_a'] + num_relatives * res_r['cost_a']
    fc_b = res_p['cost_b'] + num_relatives * res_r['cost_b']
    fq_a = res_p['qaly_a'] + num_relatives * res_r['qaly_a']
    fq_b = res_p['qaly_b'] + num_relatives * res_r['qaly_b']
    fd_c = fc_a - fc_b
    fd_q = fq_a - fq_b
    f_icer = fd_c / fd_q if fd_q != 0 else float('inf')
    df_family = pd.DataFrame({
        '対象': ['発端者 (1人)', f'血縁者 ×{num_relatives}人', '家系合計'],
        '戦略A 費用': [
            f"¥{res_p['cost_a']:,.0f}",
            f"¥{num_relatives*res_r['cost_a']:,.0f}",
            f"¥{fc_a:,.0f}"
        ],
        '戦略B 費用': [
            f"¥{res_p['cost_b']:,.0f}",
            f"¥{num_relatives*res_r['cost_b']:,.0f}",
            f"¥{fc_b:,.0f}"
        ],
        '戦略A QALY': [
            f"{res_p['qaly_a']:.4f}",
            f"{num_relatives*res_r['qaly_a']:.4f}",
            f"{fq_a:.4f}"
        ],
        '戦略B QALY': [
            f"{res_p['qaly_b']:.4f}",
            f"{num_relatives*res_r['qaly_b']:.4f}",
            f"{fq_b:.4f}"
        ],
    })
    st.table(df_family)
    st.metric("💡 家系全体 ICER",
              f"¥{f_icer:,.0f} / QALY" if np.isfinite(f_icer) else "支配的")

    ages_r = list(range(relative_age, surveillance_end_age + 1))
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(ages_r, [get_utility_norm(a) for a in ages_r], label='National norm', color='navy')
    ax2.plot(ages_r, [get_utility_norm(a)*util_healthy for a in ages_r],
             label=f'Healthy (base={util_healthy})', linestyle='--', color='green', alpha=0.7)
    ax2.set_xlabel("Age"); ax2.set_ylabel("Utility")
    ax2.set_title("Relatives: Age-adjusted Utility")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    st.pyplot(fig2); plt.close()

# ============================================================
# ---------- TAB3: TORNADO DIAGRAM (DSA) ----------
# ============================================================
with tab3:
    st.subheader("🌪️ Tornado Diagram（一変数感度分析 / DSA）")
    st.caption("各パラメータを下限・上限の間で変化させたときのICERの変動幅を示します。幅が大きいほどICERへの影響が大きいパラメータです。")

    sensitivity_params = {
        'Discount Rate':              ('discount_rate',                sa_discount_lo,           sa_discount_hi),
        'IHC Sensitivity':            ('ihc_sens',                     sa_ihc_sens_lo,           sa_ihc_sens_hi),
        'IHC Specificity':            ('ihc_spec',                     sa_ihc_spec_lo,           sa_ihc_spec_hi),
        'BRAF Sensitivity':           ('braf_sens',                    sa_braf_sens_lo,          sa_braf_sens_hi),
        'BRAF Specificity':           ('braf_spec',                    sa_braf_spec_lo,          sa_braf_spec_hi),
        'Germline Sensitivity':       ('germline_sens',                sa_germline_sens_lo,      sa_germline_sens_hi),
        'Cascade Sensitivity':        ('cascade_sens',                 sa_cascade_sens_lo,       sa_cascade_sens_hi),
        'Surveillance Uptake (Rel.)': ('surveillance_uptake_relative', sa_surv_uptake_rel_lo,    sa_surv_uptake_rel_hi),
    }

    col_t1, col_t2 = st.columns(2)
    for col_widget, label_prefix, is_prob, base_res in [
        (col_t1, "Proband", True,  res_p),
        (col_t2, "Relatives", False, res_r),
    ]:
        with col_widget:
            st.markdown(f"**{label_prefix}**")
            with st.spinner(f"{label_prefix} 感度分析中..."):
                base_icer = base_res['icer'] if np.isfinite(base_res['icer']) else 0.0
                tornado_data = []
                for pname, (pkey, lo, hi) in sensitivity_params.items():
                    icer_vals = []
                    for val in [lo, hi]:
                        pv = deepcopy(params); pv[pkey] = val
                        rv = run_cea(pv, is_proband=is_prob)
                        _fallback = 2 * 5_000_000
                        icer_vals.append(rv['icer'] if np.isfinite(rv['icer']) else _fallback)
                    tornado_data.append({
                        'param': pname,
                        'lo_param': lo, 'hi_param': hi,
                        'icer_at_lo': icer_vals[0], 'icer_at_hi': icer_vals[1],
                        'bar_left':  min(icer_vals), 'bar_right': max(icer_vals),
                        'range': abs(icer_vals[1] - icer_vals[0]),
                    })
                tdf = pd.DataFrame(tornado_data).sort_values('range', ascending=True)

                fig_t, ax_t = plt.subplots(figsize=(8, 6))
                c_lo, c_hi = '#2196F3', '#FF5722'
                for i, (_, row) in enumerate(tdf.iterrows()):
                    left, right = row['bar_left'], row['bar_right']
                    col_left  = c_lo if row['icer_at_lo'] <= row['icer_at_hi'] else c_hi
                    col_right = c_hi if row['icer_at_lo'] <= row['icer_at_hi'] else c_lo
                    ax_t.barh(i, base_icer - left, left=left, height=0.6,
                              color=col_left, alpha=0.85,
                              label='Param low' if i == 0 else "")
                    ax_t.barh(i, right - base_icer, left=base_icer, height=0.6,
                              color=col_right, alpha=0.85,
                              label='Param high' if i == 0 else "")
                    span = max(right - left, 1)
                    ax_t.text(left  - span*0.01, i, f"Y{left/1e6:.2f}M",
                              va='center', ha='right', fontsize=7)
                    ax_t.text(right + span*0.01, i, f"Y{right/1e6:.2f}M",
                              va='center', ha='left',  fontsize=7)
                ax_t.axvline(base_icer, color='black', linestyle='--', linewidth=1.5, label='Base ICER')
                ax_t.set_yticks(range(len(tdf)))
                ax_t.set_yticklabels(tdf['param'].tolist(), fontsize=9)
                ax_t.set_xlabel("ICER (JPY/QALY)")
                ax_t.set_title(f"{label_prefix} Tornado Diagram (DSA)")
                hdls, lbls = ax_t.get_legend_handles_labels()
                seen = {}
                for h, l in zip(hdls, lbls): seen.setdefault(l, h)
                ax_t.legend(seen.values(), seen.keys(), fontsize=8)
                ax_t.grid(True, alpha=0.3, axis='x')
                plt.tight_layout(); st.pyplot(fig_t); plt.close()

            with st.expander("感度分析テーブルを表示"):
                disp = tdf[['param','lo_param','hi_param','icer_at_lo','icer_at_hi','range']].copy()
                disp.columns = ['パラメータ','下限値','上限値','ICER(下限時)','ICER(上限時)','ICER幅']
                for col in ['ICER(下限時)', 'ICER(上限時)', 'ICER幅']:
                    disp[col] = disp[col].apply(lambda x: f"Y{x:,.0f}")
                st.dataframe(disp, use_container_width=True)

# ============================================================
# ---------- TAB4: PSA / CEAC / EVPI ----------
# ============================================================
with tab4:
    st.subheader("📈 Probability Sensitivity Analysis（PSA）/ CEAC / EVPI")

    wtp_threshold = st.number_input(
        "支払意思額 (WTP) 閾値 (JPY/QALY)",
        value=5_000_000, min_value=500_000, max_value=30_000_000, step=500_000
    )
    wtp_max_ceac = st.number_input(
        "CEAC / EVPI の WTP 最大値 (JPY/QALY)",
        value=10_000_000, min_value=1_000_000, max_value=50_000_000, step=1_000_000
    )

    if st.button("▶ PSA 実行（CEAC・EVPI を含む）"):
        with st.spinner(f"発端者 PSA ({simulation_runs}回) 実行中..."):
            psa_p = run_psa(params, is_proband=True,  n_runs=simulation_runs,
                            seed=random_seed, wtp=wtp_threshold)
        with st.spinner(f"血縁者 PSA ({simulation_runs}回) 実行中..."):
            psa_r = run_psa(params, is_proband=False, n_runs=simulation_runs,
                            seed=random_seed, wtp=wtp_threshold)

        wtp_range = np.linspace(0, wtp_max_ceac, 200)

        for df_psa, label in [(psa_p, "Proband"), (psa_r, "Relatives")]:
            st.markdown(f"---\n### {label}")

            st.markdown("#### ① Cost-Effectiveness Plane")
            fig_ce, ax_ce = plt.subplots(figsize=(6, 5))
            ax_ce.scatter(df_psa['delta_qaly'], df_psa['delta_cost'],
                          alpha=0.25, s=6, color='steelblue', label='PSA scatter')
            ax_ce.axhline(0, color='black', linewidth=0.8)
            ax_ce.axvline(0, color='black', linewidth=0.8)
            x_ce = np.linspace(df_psa['delta_qaly'].min(), df_psa['delta_qaly'].max(), 100)
            ax_ce.plot(x_ce, x_ce * wtp_threshold, 'r--', linewidth=1.5,
                       label=f'WTP=Y{wtp_threshold/1e6:.1f}M')

            mean_xy = np.array([df_psa['delta_qaly'].mean(), df_psa['delta_cost'].mean()])
            cov_xy  = np.cov(df_psa['delta_qaly'], df_psa['delta_cost'])
            vals, vecs = np.linalg.eigh(cov_xy)
            order = vals.argsort()[::-1]; vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            ell = Ellipse(xy=mean_xy, width=2*1.96*np.sqrt(vals[0]),
                          height=2*1.96*np.sqrt(vals[1]), angle=theta,
                          edgecolor='navy', fc='None', lw=1.5, label='95% ellipse')
            ax_ce.add_patch(ell)
            ax_ce.scatter(*mean_xy, color='navy', s=60, zorder=5, label='Mean')
            ax_ce.set_xlabel("Incremental QALY (dE)")
            ax_ce.set_ylabel("Incremental Cost JPY (dC)")
            ax_ce.set_title(f"{label} Cost-Effectiveness Plane")
            ax_ce.legend(fontsize=8); ax_ce.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_ce); plt.close()

            pct_ce = (df_psa['nmb'] > 0).mean() * 100
            valid_icer = df_psa['icer'].replace([np.inf, -np.inf], np.nan).dropna()
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1: st.metric("費用対効果的な割合 (NMB>0)", f"{pct_ce:.1f}%")
            with col_s2: st.metric("ICER 中央値", f"Y{valid_icer.median():,.0f}")
            with col_s3: st.metric("ICER 95%CI",
                                   f"Y{valid_icer.quantile(0.025):,.0f} - Y{valid_icer.quantile(0.975):,.0f}")

            st.markdown("#### ② CEAC")
            prob_ce_list = [(wt * df_psa['delta_qaly'] - df_psa['delta_cost'] > 0).mean()
                            for wt in wtp_range]
            fig_ceac, ax_ceac = plt.subplots(figsize=(7, 4))
            ax_ceac.plot(wtp_range/1e6, prob_ce_list, color='darkblue', linewidth=2,
                         label='Prob. CE (Strategy A)')
            ax_ceac.axvline(wtp_threshold/1e6, color='red', linestyle='--',
                            label=f'WTP={wtp_threshold/1e6:.1f}M')
            ax_ceac.axhline(0.5, color='gray', linestyle=':', linewidth=1)
            ax_ceac.fill_between(wtp_range/1e6, prob_ce_list, alpha=0.15, color='darkblue')
            ax_ceac.set_xlabel("Willingness-to-Pay (1.0M JPY/QALY)")
            ax_ceac.set_ylabel("Probability Cost-Effective")
            ax_ceac.set_title(f"{label} CEAC")
            ax_ceac.legend(fontsize=9); ax_ceac.grid(True, alpha=0.3); ax_ceac.set_ylim(0, 1)
            plt.tight_layout(); st.pyplot(fig_ceac); plt.close()

            st.markdown("#### ③ EVPI")
            evpi_list = []
            for wt in wtp_range:
                nmb_a_w = wt * df_psa['delta_qaly'] - df_psa['delta_cost']
                nmb_b_w = np.zeros(len(df_psa))
                evpi_w = np.maximum(nmb_a_w, nmb_b_w).mean() - max(nmb_a_w.mean(), 0.0)
                evpi_list.append(max(evpi_w, 0.0))

            evpi_at_wtp = float(np.interp(wtp_threshold, wtp_range, evpi_list))
            fig_evpi, ax_evpi = plt.subplots(figsize=(7, 4))
            ax_evpi.plot(wtp_range/1e6, [v/1e6 for v in evpi_list],
                         color='darkorange', linewidth=2, label='EVPI')
            ax_evpi.axvline(wtp_threshold/1e6, color='red', linestyle='--',
                            label=f'WTP={wtp_threshold/1e6:.1f}M')
            ax_evpi.axhline(evpi_at_wtp/1e6, color='gray', linestyle=':')
            ax_evpi.fill_between(wtp_range/1e6, [v/1e6 for v in evpi_list],
                                 alpha=0.15, color='darkorange')
            ax_evpi.set_xlabel("Willingness-to-Pay (1.0M JPY/QALY)")
            ax_evpi.set_ylabel("EVPI (1.0M JPY/person)")
            ax_evpi.set_title(f"{label} EVPI Curve")
            ax_evpi.legend(fontsize=9); ax_evpi.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_evpi); plt.close()
            st.metric(f"EVPI @ WTP=Y{wtp_threshold/1e6:.1f}M",
                      f"Y{evpi_at_wtp:,.0f} / person")

            st.markdown("#### ④ NMB Distribution")
            fig_nmb, ax_nmb = plt.subplots(figsize=(7, 3))
            ax_nmb.hist(df_psa['nmb']/1e6, bins=60, color='steelblue', alpha=0.75, edgecolor='white')
            ax_nmb.axvline(0, color='red', linewidth=1.5, linestyle='--', label='NMB=0')
            ax_nmb.axvline(df_psa['nmb'].mean()/1e6, color='navy', linewidth=1.5,
                           label=f"Mean={df_psa['nmb'].mean()/1e6:.2f}M")
            ax_nmb.set_xlabel("NMB (1.0M JPY)")
            ax_nmb.set_ylabel("Frequency")
            ax_nmb.set_title(f"{label} NMB Distribution (WTP=Y{wtp_threshold/1e6:.1f}M)")
            ax_nmb.legend(fontsize=9); ax_nmb.grid(True, alpha=0.3)
            plt.tight_layout(); st.pyplot(fig_nmb); plt.close()

    else:
        st.info("「PSA 実行」ボタンを押してください。PSA・CEAC・EVPIを一括計算します。")

# ============================================================
# ---------- TAB5: VALIDATION ----------
# ============================================================
with tab5:
    st.subheader("✅ Validation: Internal Validity Verification of Markov Models")

    CYCLE_LENGTH = 1.0

    tp_val_surv   = compute_transition_probs(params, has_surveillance=True)
    tp_val_nosurv = compute_transition_probs(params, has_surveillance=False)

    utilities_val = {
        'healthy':    util_healthy, 'adenoma':    util_adenoma,
        'early':      util_early,   'local':      util_local,
        'dfs_l':      util_dfs_l,   'metastatic': util_metastatic,
        'dfs_m':      util_dfs_m,   'bsc':        util_bsc,   'death': 0.0,
    }
    costs_val_annual = {
        'healthy':    0,
        'adenoma':    cost_colonoscopy,
        'early':      0,
        'local':      0,
        'dfs_l':      0,
        'metastatic': cost_chemo,
        'dfs_m':      0,      'bsc': cost_bsc, 'colonoscopy': cost_colonoscopy,
    }
    costs_val_entry = {
        'early': cost_endoscopic_tx,
        'local': cost_surgery,
    }

    states_plot  = ['healthy', 'adenoma', 'early', 'local',
                    'dfs_l', 'metastatic', 'dfs_m', 'bsc', 'death']
    states_alive = [s for s in states_plot if s != 'death']
    state_colors = {
        'healthy':    '#4CAF50', 'adenoma':    '#8BC34A', 'early':   '#FFC107',
        'local':      '#FF9800', 'dfs_l':      '#03A9F4', 'metastatic': '#F44336',
        'dfs_m':      '#9C27B0', 'bsc':        '#795548', 'death':   '#37474F',
    }
    state_labels = {
        'healthy':    'Healthy',          'adenoma':    'Adenoma',
        'early':      'Early(Endoscopy)', 'local':      'Local(Surg->Chemo)',
        'dfs_l':      'DFS_L',            'metastatic': 'Metastatic(Chemo)',
        'dfs_m':      'DFS_M',            'bsc':        'BSC',
        'death':      'Death',
    }

    with st.expander("Transition probability comparison"):
        ci1, ci2 = st.columns(2)
        with ci1:
            st.markdown("**Strategy A: Surveillance**")
            st.write(f"Healthy->Adenoma:    {tp_val_surv['healthy_to_adn']:.4f}")
            st.write(f"Healthy->Early:      {tp_val_surv['healthy_to_early']:.4f}")
            st.write(f"Healthy->Local:      {tp_val_surv['healthy_to_local']:.4f}")
            st.write(f"Healthy->Metastatic: {tp_val_surv['healthy_to_meta']:.4f}")
        with ci2:
            st.markdown("**Strategy B: No Surveillance**")
            st.write(f"Healthy->Adenoma:    {tp_val_nosurv['healthy_to_adn']:.4f}")
            st.write(f"Healthy->Early:      {tp_val_nosurv['healthy_to_early']:.4f}")
            st.write(f"Healthy->Local:      {tp_val_nosurv['healthy_to_local']:.4f}")
            st.write(f"Healthy->Metastatic: {tp_val_nosurv['healthy_to_meta']:.4f}")

    val_tab1, val_tab2 = st.tabs(["Proband", "Relatives"])

    for val_tab, label, start_a in [
        (val_tab1, "Proband",   proband_age),
        (val_tab2, "Relatives", relative_age),
    ]:
        with val_tab:
            _, _, trace_surv = run_markov_model(
                start_a, surveillance_end_age, discount_rate,
                tp_val_surv, utilities_val, costs_val_annual,
                'healthy', True, True,
                costs_on_entry=costs_val_entry)
            _, _, trace_nosurv = run_markov_model(
                start_a, surveillance_end_age, discount_rate,
                tp_val_nosurv, utilities_val, costs_val_annual,
                'healthy', False, True,
                costs_on_entry=costs_val_entry)

            traces = {
                "Strategy A: Surveillance":    trace_surv,
                "Strategy B: No Surveillance": trace_nosurv,
            }
            tp_map = {
                "Strategy A: Surveillance":    tp_val_surv,
                "Strategy B: No Surveillance": tp_val_nosurv,
            }

            st.markdown("### ① Stacked Area Trace Plot")
            c1, c2 = st.columns(2)
            for col_w, (slabel, trace_df) in zip([c1, c2], traces.items()):
                with col_w:
                    st.markdown(f"**{slabel}**")
                    fig, axes = plt.subplots(2, 1, figsize=(6, 8),
                                             gridspec_kw={'height_ratios': [3, 1]})
                    ys = np.array([trace_df[s].values for s in states_plot])
                    axes[0].stackplot(trace_df['age'], ys,
                                      labels=[state_labels[s] for s in states_plot],
                                      colors=[state_colors[s]  for s in states_plot],
                                      alpha=0.85)
                    axes[0].set_xlim(start_a, surveillance_end_age)
                    axes[0].set_ylim(0, 1)
                    axes[0].set_xlabel("Age"); axes[0].set_ylabel("Proportion")
                    axes[0].set_title(f"{label} - {slabel}")
                    axes[0].legend(loc='lower left', fontsize=7, ncol=2)
                    axes[0].grid(True, alpha=0.2)

                    csum = trace_df[states_plot].sum(axis=1)
                    axes[1].plot(trace_df['age'], csum, color='navy', linewidth=2)
                    axes[1].axhline(1.0, color='red', linestyle='--', linewidth=1)
                    axes[1].set_xlim(start_a, surveillance_end_age)
                    axes[1].set_ylim(0.98, 1.02)
                    axes[1].set_xlabel("Age"); axes[1].set_ylabel("Cohort Sum")
                    axes[1].set_title("Conservation Law")
                    axes[1].grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                    dev = abs(csum - 1.0).max()
                    (st.success if dev < 1e-9 else st.warning)(
                        f"{'Conservation OK' if dev < 1e-9 else 'Conservation deviation'} (max dev {dev:.2e})"
                    )

            st.markdown("### ② Expected Life Years Validation")
            c1, c2 = st.columns(2)
            for col_w, (slabel, trace_df) in zip([c1, c2], traces.items()):
                with col_w:
                    st.markdown(f"**{slabel}**")
                    ages_ly  = trace_df['age'].values
                    alive_ly = 1.0 - trace_df['death'].values
                    hcw = np.ones(len(alive_ly))
                    hcw[0] = hcw[-1] = 0.5
                    ly_model = np.cumsum(alive_ly * hcw * CYCLE_LENGTH)

                    surv_bg = np.ones(len(ages_ly))
                    for k in range(1, len(ages_ly)):
                        surv_bg[k] = surv_bg[k-1] * (1.0 - get_background_mortality(ages_ly[k-1]))
                    ly_lt = np.cumsum(surv_bg * CYCLE_LENGTH)

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(ages_ly, ly_model, color='steelblue', linewidth=2, label='Model LY')
                    ax.plot(ages_ly, ly_lt, color='gray', linewidth=1.5, linestyle='--',
                            label='Life table LY')
                    ax.fill_between(ages_ly, ly_model, ly_lt, alpha=0.15, color='steelblue')
                    ax.set_xlabel("Age"); ax.set_ylabel("Cumulative Life Years")
                    ax.set_title(f"Expected Life Years\n{label} - {slabel}")
                    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                    m_tot = ly_model[-1]; lt_tot = ly_lt[-1]
                    st.write(f"Model LY: **{m_tot:.2f}y**  Life table LY: **{lt_tot:.2f}y**  "
                             f"Disease burden: **{lt_tot - m_tot:.2f}y**")

# ============================================================
# ---------- TAB6: FLOWCHART ----------
# ============================================================
with tab6:
    st.subheader("🌳 Flowchart")

    try:
        import graphviz
        _graphviz_ok = True
    except ImportError:
        _graphviz_ok = False
        st.warning("graphviz が利用できません。`pip install graphviz` が必要です。")

    if _graphviz_ok:
        flow_tab1, flow_tab2 = st.tabs(["Decision Tree", "Markov State Diagram"])

        with flow_tab2:
            st.markdown("#### Markov State Transition Diagram")
            tp_diag_surv   = compute_transition_probs(params, has_surveillance=True)
            tp_diag_nosurv = compute_transition_probs(params, has_surveillance=False)
            _surv_mode = st.radio("Display mode", ["With Surveillance", "Without Surveillance"],
                                  horizontal=True, key="markov_flow_mode")
            _tp = tp_diag_surv if _surv_mode == "With Surveillance" else tp_diag_nosurv

            dot = graphviz.Digraph(
                graph_attr={'rankdir': 'TB', 'bgcolor': '#FAFAFA'},
                node_attr={'fontname': 'Helvetica', 'fontsize': '10'},
                edge_attr={'fontname': 'Helvetica', 'fontsize': '8'},
            )
            state_styles = {
                'healthy':    {'label': 'Healthy',      'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#C8E6C9'},
                'adenoma':    {'label': 'Adenoma',       'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#DCEDC8'},
                'early':      {'label': 'Early\n(Endo)', 'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#FFF9C4'},
                'local':      {'label': 'Local\n(Surg)', 'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#FFE0B2'},
                'dfs_l':      {'label': 'DFS_L',         'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#B3E5FC'},
                'metastatic': {'label': 'Metastatic',    'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#FFCDD2'},
                'dfs_m':      {'label': 'DFS_M',         'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#E1BEE7'},
                'bsc':        {'label': 'BSC',           'shape': 'ellipse', 'style': 'filled', 'fillcolor': '#D7CCC8'},
                'death':      {'label': 'Death',         'shape': 'doublecircle', 'style': 'filled', 'fillcolor': '#CFD8DC'},
            }
            for sid, attrs in state_styles.items():
                dot.node(sid, **attrs)

            def _edge(src, dst, prob):
                if prob > 0.0001:
                    dot.edge(src, dst, label=f' {prob:.3f} ')

            _edge('healthy', 'adenoma',    _tp['healthy_to_adn'])
            _edge('healthy', 'early',      _tp['healthy_to_early'])
            _edge('healthy', 'local',      _tp['healthy_to_local'])
            _edge('healthy', 'metastatic', _tp['healthy_to_meta'])
            _edge('adenoma', 'healthy',    _tp['adn_healthy'])
            _edge('early',   'healthy',    _tp['early_cure'])
            _edge('local',   'dfs_l',      _tp['local_dfs_l'])
            _edge('dfs_l',   'healthy',    _tp['dfsl_healthy'])
            _edge('dfs_l',   'metastatic', _tp['dfsl_meta'])
            _edge('metastatic', 'dfs_m',   _tp['meta_dfsm'])
            _edge('metastatic', 'bsc',     _tp['meta_bsc'])
            _edge('metastatic', 'healthy', _tp['meta_healthy'])
            _edge('dfs_m',   'healthy',    _tp['dfsm_healthy'])
            _edge('dfs_m',   'metastatic', _tp['dfsm_meta'])
            _edge('bsc',     'death',      _tp['bsc_death'])
            st.graphviz_chart(dot, use_container_width=True)

        with flow_tab1:
            st.markdown("#### Diagnostic Flowchart")
            p = params
            _N = st.number_input("Cohort size N", value=100_000, min_value=1000, max_value=1_000_000, step=1000)

            _lynch_prev    = p['lynch_prevalence']
            _mmr_pos_rate  = p['mmr_deficient_rate']
            _ihc_sens      = p['ihc_sens']
            _ihc_spec      = p['ihc_spec']
            _mlh1_neg_rate = p['mlh1_rate']
            _braf_sens     = p['braf_sens']
            _braf_spec     = p['braf_spec']
            _gc_uptake     = p['gc_uptake_proband']
            _gt_uptake     = p['gt_uptake_proband']
            _germline_sens = p['germline_sens']
            _n_relatives   = p['num_relatives']
            _gc_uptake_rel = p['gc_uptake_relative']
            _gt_uptake_rel = p['gt_uptake_relative']
            _carrier_rate  = p['relative_carrier_rate']
            _cascade_sens  = p['cascade_sens']

            _ihc_tp_n   = _N * _ihc_sens * _mmr_pos_rate
            _ihc_fp_n   = _N * (1.0 - _ihc_spec) * (1.0 - _mmr_pos_rate)
            _pmmr_n     = _N * _ihc_spec * (1.0 - _mmr_pos_rate)
            _ihc_fn_n   = _N * (1.0 - _ihc_sens) * _mmr_pos_rate
            _ihc_pos_n  = _ihc_tp_n + _ihc_fp_n
            _mlh1_neg_n = _ihc_pos_n * _mlh1_neg_rate
            _others_pos_n = _ihc_pos_n * (1.0 - _mlh1_neg_rate)
            _braf_tested_n = _mlh1_neg_n
            _braf_lynch_in = _ihc_sens * _lynch_prev * _N * _mlh1_neg_rate
            _braf_nonlynch_in = _braf_tested_n - _braf_lynch_in
            _braf_pos_n = _braf_nonlynch_in * _braf_sens
            _braf_fp_lynch = _braf_lynch_in * (1.0 - _braf_spec)
            _braf_neg_nonlynch = _braf_nonlynch_in * (1.0 - _braf_sens)
            _braf_neg_lynch    = _braf_lynch_in * _braf_spec
            _braf_neg_n = _braf_neg_nonlynch + _braf_neg_lynch
            _gc_candidates_n = _braf_neg_n + _others_pos_n
            _n_counseled = _gc_candidates_n * _gc_uptake
            _n_tested    = _n_counseled * _gt_uptake
            _lynch_in_gc = _braf_neg_lynch + _ihc_sens * _lynch_prev * _N * (1.0 - _mlh1_neg_rate)
            _lynch_frac  = float(np.clip(_lynch_in_gc / max(_gc_candidates_n, 1e-10), 0.0, 1.0))
            _n_ls_diagnosed = _n_tested * _germline_sens * _lynch_frac
            _n_no_gpv = _n_tested - _n_ls_diagnosed
            _n_relatives_cand = _n_ls_diagnosed * _n_relatives
            _n_rel_counseled  = _n_relatives_cand * _gc_uptake_rel
            _n_rel_tested     = _n_rel_counseled * _gt_uptake_rel
            _total_rel_found  = _n_rel_tested * _cascade_sens * _carrier_rate

            graph = f"""
digraph {{
    rankdir=TB;
    node [shape=box, style="rounded,filled", fillcolor="white", fontname="Helvetica"];
    edge [fontname="Helvetica", fontsize=10];
    Start      [label="CRC Patients\\nN = {int(_N):,}", fillcolor="lightyellow"];
    IHC        [label="MMR-IHC\\nPos: {int(_ihc_pos_n):,}", shape=diamond, fillcolor="lightyellow"];
    BRAF       [label="BRAF\\nN = {int(_braf_tested_n):,}", shape=diamond, fillcolor="lightyellow"];
    Candidates [label="GC Candidates\\nN = {int(_gc_candidates_n):,}", fillcolor="lavender"];
    Counseling [label="Genetic Counseling\\nN = {int(_n_counseled):,}", fillcolor="lavender"];
    Testing    [label="Germline Testing\\nN = {int(_n_tested):,}", fillcolor="lavender"];
    LS_Confirmed [label="Lynch Syndrome\\nN = {int(_n_ls_diagnosed):,}", fillcolor="gold"];
    Rel_Start  [label="Relatives\\nN = {int(_n_relatives_cand):,}", fillcolor="lightcyan"];
    Rel_Coun   [label="Cascade Counseling\\nN = {int(_n_rel_counseled):,}", fillcolor="lightcyan"];
    Rel_Test   [label="Cascade Testing\\nN = {int(_n_rel_tested):,}", fillcolor="lightcyan"];
    Rel_Found  [label="LS Relatives Found\\nN = {int(_total_rel_found):,}", fillcolor="salmon"];
    Start -> IHC;
    IHC -> BRAF [label="MLH1 neg"];
    IHC -> Candidates [label="Other MMR"];
    BRAF -> Candidates [label="BRAF neg"];
    Candidates -> Counseling [label="{_gc_uptake*100:.0f}%"];
    Counseling -> Testing [label="{_gt_uptake*100:.0f}%"];
    Testing -> LS_Confirmed [label="GPV+"];
    LS_Confirmed -> Rel_Start [label="x{_n_relatives} relatives"];
    Rel_Start -> Rel_Coun [label="{_gc_uptake_rel*100:.0f}%"];
    Rel_Coun -> Rel_Test [label="{_gt_uptake_rel*100:.0f}%"];
    Rel_Test -> Rel_Found [label="{_carrier_rate*100:.0f}%"];
}}
"""
            st.graphviz_chart(graph, use_container_width=False)

# ============================================================
# ---------- TAB7: PARAMETERS ----------
# ============================================================
with tab7:
    st.subheader("📋 Parameter Summary")
    st.caption("All parameters currently set in the sidebar.")

    st.markdown("### Common Settings")
    df_common = pd.DataFrame({
        "Parameter": ["Discount Rate", "Proband Start Age", "Relative Start Age",
                      "Surveillance End Age", "MC Runs", "Random Seed"],
        "Value": [f"{discount_rate:.3f}", str(proband_age), str(relative_age),
                  str(surveillance_end_age), str(simulation_runs), str(random_seed)],
    })
    st.dataframe(df_common, use_container_width=True, hide_index=True)

    st.markdown("### Diagnostic Accuracy")
    df_diag = pd.DataFrame({
        "Test": ["IHC", "BRAF", "Germline", "Cascade"],
        "Sensitivity": [f"{ihc_sens:.2f}", f"{braf_sens:.2f}", f"{germline_sens:.2f}", f"{cascade_sens:.2f}"],
        "Specificity": [f"{ihc_spec:.2f}", f"{braf_spec:.2f}", f"{germline_spec:.2f}", f"{cascade_spec:.2f}"],
    })
    st.dataframe(df_diag, use_container_width=True, hide_index=True)

    st.info("💡 Sidebar changes update this table in real time.")

with tab8:
    st.subheader("📚 References & Notes")
    """
    #### Prevalence
    ##### General Population
    * [Cancer Epidemiol Biomarkers Prev. 2017;26(3):404-412.](https://aacrjournals.org/cebp/article/26/3/404/71285/Prevalence-and-Penetrance-of-Major-Genes-and)
    * [BBJ. Clin Gastroenterol Hepatol. 2022;20(9):2132-2141.e9.](https://www.cghjournal.org/action/showPdf?pii=S1542-3565%2820%2931664-5)
    * [BBJ_Lynch. Commun Med (Lond). 2025;5(1):522.](https://www.nature.com/articles/s43856-025-01231-9)
    * [ToMMo. Jpn J Clin Oncol. 2026;56(2):168-179.](https://pubmed.ncbi.nlm.nih.gov/41342546/)
    ##### CRC
    * [Saitama. Int J Clin Oncol. 2023;28(12):1633-1640.](https://sci-hub.st/10.1007/s10147-023-02412-z)
    * [Komagome. Int J Clin Oncol. 2024;29(7):944-952.](https://sci-hub.st/10.1007/s10147-024-02527-x)

    #### Testing Sensitivity/Specificity
    * [Gastroenterology. 2015;149(3):783-813.e20.](https://www.cghjournal.org/action/showPdf?pii=S1542-3565%2819%2931165-6)

    #### Cost
    * [J Gastroenterol. 2021;56(10):903-913.](https://sci-hub.st/10.1007/s00535-021-01798-9)
    * [Clin Gastroenterol Hepatol. 2020;18(12):2760-2767.e12.](https://www.cghjournal.org/action/showPdf?pii=S1542-3565%2819%2931165-6)

    #### Utility
    * Health Technol Assess. 2020;24(66):1-332.
    * Saudi J Gastroenterol. 2021;27(4):208-216.
    * [NHS. Health Technol Assess. 2014;18(58):1-406.](https://www.ncbi.nlm.nih.gov/books/NBK262562/)
    """

st.markdown("---")
st.caption("© 2026 Lynch Syndrome CEA Model | Powered by Streamlit + stlite")
