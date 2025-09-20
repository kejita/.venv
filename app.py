# app.py
import io
import time
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------
# 基本設定
# ------------------------------
st.set_page_config(
    page_title="ML Predictor",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 ML Predictor (Streamlit)")
st.caption("学習済みモデルを使って、単発入力 or CSV 一括で予測します。")

# ------------------------------
# モデル読込（キャッシュ）
# ------------------------------
@st.cache_resource
def load_model(path: str = "titanic_SVC_linear.ipynb"):
    model = joblib.load(path)
    return model

# ------------------------------
# 前処理（必要ならここで実装）
# ------------------------------
@st.cache_data
def preprocess_df(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    # 実案件では欠損補完や型変換、エンコーディングなどをここに
    # ここでは最低限：必要列の抽出＆欠損を落とす
    out = df.copy()
    out = out[feature_cols]
    out = out.dropna()
    return out

# ------------------------------
# 推論関数
# ------------------------------
def predict_array(model, arr: np.ndarray) -> np.ndarray:
    # scikit-learn 互換モデル想定
    return model.predict(arr)

# ------------------------------
# サイドバー：ヘルプ & 設定
# ------------------------------
with st.sidebar:
    st.header("設定 / Help")
    st.markdown(
        "- 左のフォームで単発推論\n"
        "- 下のアップローダでCSV一括推論\n"
        "- 列名は `x1, x2` を例にしています（変更OK）"
    )
    st.divider()
    # 特徴量名の設定（実案件に合わせて編集）
    x1_name = st.text_input("特徴量1の列名", value="x1")
    x2_name = st.text_input("特徴量2の列名", value="x2")
    feature_cols = [x1_name, x2_name]

# ------------------------------
# モデルのロード
# ------------------------------
try:
    model = load_model("model.pkl")
    st.success("モデルをロードしました: model.pkl")
except Exception as e:
    st.error(f"モデルのロードに失敗しました: {e}")
    st.stop()

# ------------------------------
# 単発推論フォーム
# ------------------------------
st.subheader("📝 単発入力で予測")
with st.form(key="single_predict"):
    c1, c2 = st.columns(2)
    with c1:
        v1 = st.number_input(f"{x1_name}", value=0.0, step=0.1)
    with c2:
        v2 = st.number_input(f"{x2_name}", value=0.0, step=0.1)
    submitted = st.form_submit_button("予測する")
    if submitted:
        with st.spinner("推論中..."):
            arr = np.array([[v1, v2]], dtype=float)
            y_pred = predict_array(model, arr)[0]
            time.sleep(0.2)  # 演出
        st.success(f"予測値: **{y_pred:.4f}**")

# ------------------------------
# CSV 一括推論
# ------------------------------
st.subheader("📄 CSV をアップロードして一括予測")
uploaded = st.file_uploader("CSVファイルを選択（UTF-8推奨）", type=["csv"])
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except UnicodeDecodeError:
        # 文字コード違いの救済（Shift-JIS 試行）
        uploaded.seek(0)
        df_raw = pd.read_csv(uploaded, encoding="cp932")

    st.write("アップロードデータ（先頭5行）")
    st.dataframe(df_raw.head())

    if all(col in df_raw.columns for col in feature_cols):
        df = preprocess_df(df_raw, feature_cols)
        st.info(f"前処理後の件数: {len(df)}")
        if len(df) > 0:
            with st.spinner("一括推論中..."):
                preds = predict_array(model, df[feature_cols].to_numpy())
                df_out = df.copy()
                df_out["prediction"] = preds
                time.sleep(0.3)

            st.success("推論完了！")
            st.dataframe(df_out.head())

            # ざっくり分布可視化
            fig = px.histogram(df_out, x="prediction", nbins=30, title="Prediction Distribution")
            st.plotly_chart(fig, use_container_width=True)

            # ダウンロード
            csv_bytes = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="結果をCSVでダウンロード",
                data=csv_bytes,
                file_name="predictions.csv",
                mime="text/csv"
            )
    else:
        missing = [c for c in feature_cols if c not in df_raw.columns]
        st.error(f"必要な列が見つかりません: {missing}")
        st.caption("サイドバーで列名を合わせるか、CSVの列名を調整してください。")

st.divider()
st.caption("© Streamlit Demo — 前処理/特徴量名/可視化は実案件に合わせて自由に拡張してください。")
