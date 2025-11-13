from ultralytics import YOLO

# モデルの定義（最初は軽いモデル yolov8n）
model = YOLO("yolov8n.yaml")

# 学習
model.train(
    data="C:/Users/admin/OneDrive/デスクトップ/streamlit_app/mouse_nose_dataset/data.yaml",
    epochs=50,          # 学習回数（必要に応じて増やす）
    imgsz=640,          # 画像サイズ
    batch=16,           # CPUの場合は8〜16程度がおすすめ
    name="train_debug", # 出力フォルダ名
)
