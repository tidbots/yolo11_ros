# YOLO11 ROS1 Noetic Integration (Detection2DArray + Semantic Filter)
- YOLO11（Ultralytics）を ROS1 Noetic 上で動作
- Detection2DArray を出力
- Semantic Filterを実装

## 全体構成
```
USB Camera
   ↓  /usb_cam/image_raw (sensor_msgs/Image)
yolo11_node.py
   ├─ /yolo11/debug_image        (sensor_msgs/Image)
   └─ /yolo11/detections         (vision_msgs/Detection2DArray)
            ↓
semantic_filter_node.py
            ↓
   /yolo11/filtered_detections   (vision_msgs/Detection2DArray)
```
YOLO 推論ノードとフィルタノードは完全分離

## 対応環境
- Ubuntu 20.04
- ROS1 Noetic
- Python3
- Ultralytics YOLO11
- vision_msgs (ROS1 Noetic 版)

## yolo11_node.py
### 入力
```
/usb_cam/image_raw
(sensor_msgs/Image)
```
### 出力
1. デバッグ用（人が見る）
```
/yolo11/debug_image
(sensor_msgs/Image)
```
results.plot() による bbox + class + score 描画済み画像

2. RoboCup / 把持用（ロボットが使う）
```
/yolo11/detections
(vision_msgs/Detection2DArray)
```
各 detection には
- bbox center / size（pixel）
- ObjectHypothesisWithPose
  - id : YOLO class index（int）
  - score : confidence
  - pose : identity（必須なのでダミーで設定）
 
## semantic_filter_node.py 
入力
```
/yolo11/detections
(vision_msgs/Detection2DArray)
```
出力
```
/yolo11/filtered_detections
(vision_msgs/Detection2DArray)
```
フィルタ条件
- allow_ids : 許可する YOLO class index のリスト
- min_score : 最低 confidence

例（YOLO COCO）
```
物体	    ID
person	 0
bottle	39
cup	    41
```

## launch 設定例
bottle のみ（把持用）
```
<node pkg="yolo11_ros"
      type="semantic_filter_node.py"
      name="semantic_filter_bottle"
      output="screen">
  <param name="allow_ids" value="[39]"/>
  <param name="min_score" value="0.4"/>
</node>
```
person のみ（GPSR / HRI）
```
<param name="allow_ids" value="[0]"/>
```
## 使い方
### インストール
```
git clone https://github.com/tidbots/yolo11_ros.git
cd yolo11_ros
docker compose build
```
### 実行
```
docker compose up
```
別のターミナルで
```
docker compose exec yolo11_noetic bash
```
```
rqt_image_view
```
別のターミナルで
```
docker compose exec yolo11_noetic bash
```
```
rostopic echo /
```


