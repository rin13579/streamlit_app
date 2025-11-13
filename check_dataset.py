# check_dataset.py
from ultralytics.utils.checks import check_yaml

# Ultralyticsのバージョン差異に対応（8.2系/8.3系など）
try:
    from ultralytics.data.utils import check_det_dataset  # 新しめ
except Exception:
    from ultralytics.data.utils import check_dataset as check_det_dataset  # 互換

DATA = r"C:\Users\admin\OneDrive\デスクトップ\streamlit_app\mouse_nose_dataset\data.yaml"

data = check_yaml(DATA)          # YAMLの解決（相対/絶対を正規化）
info = check_det_dataset(data)   # パス・枚数・重複などを検証

print("✅ Dataset OK")
print(info)  # クラス数やパス情報などの要約が表示されます
