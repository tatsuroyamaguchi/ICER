# Lynch症候群 費用対効果分析 (CEA)

## GitHub Pages でのデプロイ（推奨）

**stlite**（Streamlit の WebAssembly 版）を使うことで、
GitHub Pages 上でサーバーなしに Python / Streamlit アプリをブラウザで動かせます。

### 手順

1. このリポジトリを GitHub に push する
   ```
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/tatsuroyamaguchi/ICER.git
   git push -u origin main
   ```

2. GitHub リポジトリの **Settings → Pages** を開く

3. **Source** を `Deploy from a branch` → `main` ブランチ / `/ (root)` に設定して **Save**

4. 数分後に表示される URL（`https://tatsuroyamaguchi.github.io/ICER/`）にアクセスする

### ファイル構成

```
index.html      ← GitHub Pages エントリポイント（stlite ローダー）
app.py          ← Streamlit アプリ本体（stlite 対応修正済み）
cea_core.py     ← 計算エンジン（変更なし）
requirements.txt← Streamlit Cloud 用（GitHub Pages では不要）
```

### 修正内容（元ファイルからの変更点）

| 修正箇所 | 理由 |
|---|---|
| `st.set_page_config()` をファイル先頭に移動 | Streamlit の仕様：最初の `st` コールでなければエラー |
| `import japanize_matplotlib` を `try/except` でラップ | Pyodide（stlite）に japanize_matplotlib が非収録 |
| `import sys, os` と `sys.path.insert(...)` を削除 | stlite ではファイルをメモリにマウントするため不要 |
| `¥` 記号を `Y` に変換（グラフラベルのみ） | Pyodide の DejaVu Sans フォントが円記号を描画できないため |

### 注意事項

- **初回読み込みに 30〜60 秒**かかります（Pyodide + パッケージのダウンロード）
- 日本語フォントは `japanize_matplotlib` の代わりに `DejaVu Sans` フォールバックを使用するため、  
  グラフの日本語ラベルは一部文字化けすることがあります（UI 上のテキストは正常）
- `graphviz` は Pyodide 環境で `pip install graphviz` が必要です（Tab6 の Flowchart）

---

## Streamlit Community Cloud でのデプロイ（代替）

日本語フォントも完全に使いたい場合は Streamlit Community Cloud が推奨です。

1. [share.streamlit.io](https://share.streamlit.io) にアクセス
2. GitHub アカウントでログイン
3. **New app** → リポジトリ・ブランチを選択 → **Main file path**: `app.py` → **Deploy**

`requirements.txt` に従って自動的に依存パッケージがインストールされます。
