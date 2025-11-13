import streamlit as st
import tempfile
import cv2
import numpy as np
import pandas as pd
import os
import time
from ultralytics import YOLO
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="鼻トラッカー", layout="wide")
st.title("🐭 鼻トラッカー & 行動量解析アプリ（精度強化版）")

# -----------------------------------
# ユーティリティ
# -----------------------------------
def filter_nose_boxes(boxes, frame_w, frame_h,
                      roi_y_max_pct, min_area_pct, max_area_pct,
                      min_ar, max_ar):
    """
    boxes: Ultralytics Boxes
    返り値: [(x,y,conf,w,h), ...]（条件を満たす候補のみ）
    """
    if boxes is None or boxes.xywh is None:
        return []

    y_max_abs = roi_y_max_pct * frame_h  # このYより下は捨てる
    min_area_abs = min_area_pct * (frame_w * frame_h)
    max_area_abs = max_area_pct * (frame_w * frame_h)

    kept = []
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i].item())
        if cls_id != 0:  # 鼻クラスのみ
            continue
        x = float(boxes.xywh[i][0].item())
        y = float(boxes.xywh[i][1].item())
        w = float(boxes.xywh[i][2].item())
        h = float(boxes.xywh[i][3].item())
        conf = float(boxes.conf[i].item())

        # ROI: 下側（足が写りがち）を無視
        if y > y_max_abs:
            continue

        area = w * h
        if not (min_area_abs <= area <= max_area_abs):
            continue

        ar = w / (h + 1e-6)
        if not (min_ar <= ar <= max_ar):
            continue

        kept.append((x, y, conf, w, h))
    return kept


def ema(prev_pt, curr_pt, alpha=0.5):
    """指数移動平均で位置をなめらかにする"""
    if prev_pt is None:
        return curr_pt
    px, py = prev_pt
    cx, cy = curr_pt
    return (alpha * cx + (1 - alpha) * px, alpha * cy + (1 - alpha) * py)


# -----------------------------------
# タブの作成
# -----------------------------------
tabs = st.tabs(["📥 アップロード", "📊 解析結果", "🖼️ 画像推論", "⚙️ 設定", "🧪 開発中"])

DEFAULT_MODEL_PATH = "runs/detect/train_debug/weights/best.pt"

# -----------------------------------
# 📥 アップロードタブ
# -----------------------------------
with tabs[0]:
    st.header("動画のアップロード")
    uploaded_file = st.file_uploader("動画をアップロード", type=["mp4", "mov", "avi", "mts"])

    if uploaded_file:
        st.video(uploaded_file)
        st.success("アップロード完了！")

        # 一時ファイルに保存
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.read())
            video_path = tmp.name

        # 画像抽出（1秒毎）
        if st.button("この動画から1秒ごとに画像抽出"):
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps))
            frame_count = 0
            saved_count = 0

            base_name = os.path.splitext(uploaded_file.name)[0]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_dir = f"mouse_nose_dataset/images/train/{base_name}_{timestamp}"
            os.makedirs(output_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count % frame_interval == 0:
                    filename = f"frame_{saved_count:04}.jpg"
                    filepath = os.path.join(output_dir, filename)
                    cv2.imwrite(filepath, frame)
                    saved_count += 1
                frame_count += 1

            cap.release()
            st.success(f"{saved_count} 枚の画像を保存しました！保存先: {output_dir}")

        st.subheader("鼻先検出と移動距離解析（精度チューニング付き）")

        # モデル
        model_path_ui = st.text_input("学習済みモデルのパス", value=DEFAULT_MODEL_PATH)
        model = YOLO(model_path_ui)
        st.caption(f"モデルのクラス名: {getattr(model, 'names', {0: 'nose'})}")

        # 推論・制約パラメータ
        left, right = st.columns(2)
        with left:
            base_conf = st.slider("基本confしきい値", 0.01, 0.70, 0.20, 0.01)
            min_conf = st.slider("最小conf（見失い時に一時的に下げる下限）", 0.01, 0.70, 0.10, 0.01)
            imgsz = st.selectbox("推論解像度 (imgsz)", [320, 480, 640, 800], index=2)
            ema_alpha = st.slider("平滑化(EMA) α", 0.05, 0.95, 0.40, 0.05)
            patience_frames = st.number_input("検出が消えても位置保持するフレーム数", 0, 30, 5, 1)
            max_jump_px = st.number_input("1フレームの最大許容移動量(px)", 1, 500, 60, 1)
        with right:
            roi_y_max_pct = st.slider("有効ROIの高さ(上からの割合) ※下側無視", 0.10, 1.00, 0.80, 0.05)
            min_area_pct = st.slider("最小ボックス面積(画素比)", 0.00001, 0.01, 0.0002, 0.00001, format="%.5f")
            max_area_pct = st.slider("最大ボックス面積(画素比)", 0.001, 0.20, 0.02, 0.001, format="%.3f")
            min_ar = st.slider("最小アスペクト比 w/h", 0.10, 2.00, 0.50, 0.05)
            max_ar = st.slider("最大アスペクト比 w/h", 0.50, 4.00, 2.00, 0.05)
            show_preview = st.checkbox("デバッグ用プレビュー表示（遅くなります）", value=False)

        # スケール
        pixels_per_cm = float(st.session_state.get("pixels_per_cm", 79.1))
        threshold_px = float(st.session_state.get("threshold", 2.0))

        # 解析ループ
        cap = cv2.VideoCapture(video_path)
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        st.write("解析中... お待ちください")

        # 軌跡動画出力
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_path = "trajectory_output.mp4"
        out = cv2.VideoWriter(out_path, fourcc, max(1.0, fps_in), (frame_w, frame_h))

        prev_nose = None          # EMA後の位置
        last_det = None           # 直近の「検出された」位置（EMA前後どちらでもOK）
        no_det_count = 0
        total_distance = 0.0
        frame_distances, frame_ids = [], []
        frame_idx = 0
        conf_curr = base_conf     # 適応conf

        trajectory_points = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 推論
            results = model.predict(source=frame, conf=conf_curr, imgsz=imgsz, verbose=False)
            r0 = results[0]
            candidates = filter_nose_boxes(
                r0.boxes, frame_w, frame_h,
                roi_y_max_pct, min_area_pct, max_area_pct,
                min_ar, max_ar
            )

            # 最良（conf最大）候補
            if len(candidates) > 0:
                x, y, conf, bw, bh = max(candidates, key=lambda t: t[2])
                curr_det = (x, y)
                last_det = curr_det
                no_det_count = 0
                # 検出できたらconfをベース値に戻す
                conf_curr = base_conf
            else:
                curr_det = None
                no_det_count += 1
                # 見失い時は一時的にしきい値を下げて再探索
                if no_det_count <= patience_frames:
                    # 直近位置を保持（距離は後で判定）
                    pass
                else:
                    # さらに見失いが続くなら、confを少しずつ min_conf まで下げる
                    conf_curr = max(min_conf, conf_curr - 0.05)

            # 位置決定：検出なければ last_det を仮位置にしてEMA
            raw_target = curr_det if curr_det is not None else last_det
            if raw_target is not None:
                smoothed = ema(prev_nose, raw_target, alpha=ema_alpha)
            else:
                smoothed = None

            # 距離加算（ジャンプ抑制＆最小距離）
            if smoothed is not None and prev_nose is not None:
                dist_px = float(np.linalg.norm(np.array(smoothed) - np.array(prev_nose)))
                if dist_px <= max_jump_px and dist_px >= threshold_px:
                    total_distance += dist_px
                    frame_distances.append(total_distance / pixels_per_cm)
                    frame_ids.append(frame_idx)

            # 軌跡描画＆プレビュー
            vis = frame.copy()
            if smoothed is not None:
                trajectory_points.append(smoothed)
                cv2.circle(vis, (int(smoothed[0]), int(smoothed[1])), 6, (0, 255, 0), -1)
            # 軌跡線
            for i in range(1, len(trajectory_points)):
                pt1 = (int(trajectory_points[i-1][0]), int(trajectory_points[i-1][1]))
                pt2 = (int(trajectory_points[i][0]), int(trajectory_points[i][1]))
                cv2.line(vis, pt1, pt2, (0, 255, 255), 2)

            # ROI可視化（下側をマスクしていることを示す）
            roi_line_y = int(roi_y_max_pct * frame_h)
            cv2.line(vis, (0, roi_line_y), (frame_w, roi_line_y), (255, 0, 0), 1)

            # 出力動画へ
            out.write(vis)

            if show_preview:
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Frame {frame_idx} preview", use_column_width=True)

            prev_nose = smoothed if smoothed is not None else prev_nose
            frame_idx += 1

        cap.release()
        out.release()

        # 結果保存
        total_cm = total_distance / pixels_per_cm
        df = pd.DataFrame({"Frame": frame_ids, "Distance_cm": frame_distances})
        st.session_state["result_df"] = df
        st.session_state["total_cm"] = total_cm

        st.success(f"総移動距離: {total_cm:.2f} cm")
        st.subheader("🎬 軌跡つき動画プレビュー")
        st.video(out_path)
        st.download_button("軌跡動画をダウンロード", data=open(out_path, "rb").read(),
                           file_name="trajectory_output.mp4", mime="video/mp4")

# -----------------------------------
# 📊 解析結果タブ
# -----------------------------------
with tabs[1]:
    st.header("解析結果")
    if "result_df" in st.session_state:
        df = st.session_state["result_df"]
        fig, ax = plt.subplots()
        ax.plot(df["Frame"], df["Distance_cm"], label="累積距離(cm)")
        ax.set_xlabel("フレーム")
        ax.set_ylabel("距離(cm)")
        ax.set_title("移動距離の推移")
        ax.legend()
        st.pyplot(fig)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("CSVをダウンロード", csv, file_name="distance_data.csv", mime="text/csv")
        st.metric("総移動距離", f"{st.session_state['total_cm']:.2f} cm")
    else:
        st.info("📥 タブから動画をアップして解析を行ってください。")

# -----------------------------------
# 🖼️ 画像推論タブ
# -----------------------------------
with tabs[2]:
    st.header("画像からの鼻先検出")
    model_path_img = st.text_input("学習済みモデルのパス（画像用）", value=DEFAULT_MODEL_PATH)
    model_img = YOLO(model_path_img)
    image_file = st.file_uploader("画像をアップロード", type=["jpg", "jpeg", "png"])

    if image_file is not None:
        image = Image.open(image_file)
        st.image(image, caption="アップロードされた画像", use_column_width=True)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name

        results = model_img.predict(tmp_path, save=False, conf=0.25)
        result_img = results[0].plot()
        st.image(result_img, caption="検出結果", use_column_width=True)

        st.subheader("検出されたボックス情報")
        for box in results[0].boxes.xyxy.cpu().numpy():
            st.write(f"X1: {box[0]:.2f}, Y1: {box[1]:.2f}, X2: {box[2]:.2f}, Y2: {box[3]:.2f}")

# -----------------------------------
# ⚙️ 設定タブ
# -----------------------------------
with tabs[3]:
    st.header("設定")
    st.session_state["pixels_per_cm"] = st.number_input("1cmあたりのピクセル数", value=79.1, step=0.1)
    st.session_state["threshold"] = st.number_input("移動と判定する最小距離（px）", value=2.0, step=0.1)

# -----------------------------------
# 🧪 開発中タブ
# -----------------------------------
with tabs[4]:
    st.header("開発中機能")
    st.warning("今後ここに、新たな機能を追加予定です！")



    