from flask import Flask, render_template, request, jsonify, url_for
from ultralytics import YOLO
import os
import cv2
import numpy as np
from deep_sort.deep_sort import DeepSort
from deep_sort.deep_sort_folder.tools import xyxy_to_xywh
from collections import defaultdict
from werkzeug.utils import secure_filename
import imageio_ffmpeg as ffmpeg
import subprocess
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import multiprocessing
multiprocessing.set_start_method('spawn', force=True)

def convert_mp4_for_browser(input_path, output_path):
    ffmpeg_bin = ffmpeg.get_ffmpeg_exe()
    subprocess.call([
        ffmpeg_bin, "-y", "-i", input_path,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-movflags", "+faststart",
        output_path
    ])

def save_heatmap(points, width, height, output_path):
    if not points:
        return
    heatmap, _, _ = np.histogram2d(
        [p[0] for p in points],
        [p[1] for p in points],
        bins=[width // 10, height // 10],
        range=[[0, width], [0, height]]
    )
    heatmap = np.rot90(heatmap)
    heatmap = np.flipud(heatmap)
    plt.imshow(heatmap, cmap='jet', interpolation='nearest', alpha=0.8)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Model ve DeepSORT başlat
model = YOLO("best_optimazed.pt")
deepsort = DeepSort(model_path="deep_sort/deep/checkpoint/ckpt.t7", use_cuda=False)

# Modelden sınıf isimlerini al
ALLOWED_CLASSES = [model.names[i] for i in sorted(model.names)]

# Renk eşlemesi: model sırasına göre
CLASS_COLORS = {
    'person': (255, 0, 0),
    'bicycle': (0, 255, 255),
    'car': (0, 255, 0),
    'motorcycle': (255, 255, 0),
    'bus': (255, 0, 255),
    'truck': (0, 0, 255),
}

CONF_THRESHOLD = 0.4
id2label = {}
track_times = {}
heatmap_points = []

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/analyze', methods=['POST'])
def analyze():
    video = request.files.get("video")
    if not video:
        return jsonify({"error": "No video uploaded"}), 400

    filename = secure_filename(video.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    raw_output_path = os.path.join(UPLOAD_FOLDER, f"tracked_{filename}")
    final_output_path = raw_output_path.replace(".mp4", "_final.mp4")
    heatmap_path = os.path.join(UPLOAD_FOLDER, f"heatmap_{filename.replace('.mp4', '.png')}")
    video.save(input_path)

    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS) or 25)

    out = cv2.VideoWriter(raw_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    object_counts = defaultdict(set)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = results.boxes.data.cpu().numpy() if results.boxes.data is not None else []

        bbox_xyxy = []
        confs = []
        clss = []

        for *xyxy, conf, cls in detections:
            if conf < CONF_THRESHOLD:
                continue
            label = model.names[int(cls)]
            if label not in ALLOWED_CLASSES:
                continue
            bbox_xyxy.append(xyxy)
            confs.append(float(conf))
            clss.append(int(cls))

        outputs = []
        if bbox_xyxy:
            bbox_xywh = [xyxy_to_xywh(box) for box in bbox_xyxy]
            outputs = deepsort.update(np.array(bbox_xywh), np.array(confs), frame)

        for i, det in enumerate(bbox_xywh if bbox_xyxy else []):
            x_center, y_center, _, _ = det
            for output in outputs:
                x1, y1, x2, y2, track_id = map(int, output)
                if x1 <= x_center <= x2 and y1 <= y_center <= y2:
                    cls = clss[i]
                    label = model.names[int(cls)]
                    conf = confs[i]

                    if track_id not in id2label:
                        id2label[track_id] = label
                    else:
                        current_label = id2label[track_id]
                        if label != current_label and conf > 0.55:
                            id2label[track_id] = label
                            if track_id in object_counts[current_label]:
                                object_counts[current_label].remove(track_id)

                    true_label = id2label[track_id]
                    color = CLASS_COLORS.get(true_label, (255, 255, 255))
                    object_counts[true_label].add(track_id)

                    if track_id not in track_times:
                        track_times[track_id] = {
                            "label": true_label,
                            "start_frame": frame_id,
                            "end_frame": frame_id
                        }
                    else:
                        track_times[track_id]["end_frame"] = frame_id

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    heatmap_points.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    text = f"{true_label} {track_id}"
                    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
                    cv2.putText(frame, text, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                    break

        out.write(frame)
        frame_id += 1

    cap.release()
    out.release()
    convert_mp4_for_browser(raw_output_path, final_output_path)
    save_heatmap(heatmap_points, width, height, heatmap_path)

    counts = {label: len(ids) for label, ids in object_counts.items() if len(ids) > 0}

    for tid in track_times:
        track_times[tid]["start_sec"] = round(track_times[tid]["start_frame"] / fps, 2)
        track_times[tid]["end_sec"] = round(track_times[tid]["end_frame"] / fps, 2)

    return jsonify({
        "video_url": url_for("static", filename=f"uploads/{os.path.basename(final_output_path)}"),
        "totals": counts,
        "timeline": track_times,
        "heatmap_url": url_for("static", filename=f"uploads/{os.path.basename(heatmap_path)}")
    })

if __name__ == '__main__':
    app.run(debug=True)
