import random
import os
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
from google.colab import files
class ObjectTracker:
    def __init__(self, model, target_classes, max_age=5, gpu=False):
        self.model = model
        self.tracker = DeepSort(max_age=max_age, embedder_gpu=gpu)
        self.target_ids = self._resolve_target_ids(target_classes)
        self.color_palette = {}

    def _resolve_target_ids(self, target_classes):
        targets = [t.lower() for t in target_classes]
        return [cid for cid, name in self.model.names.items() if name.lower() in targets]

    def _get_color(self, track_id):
        if track_id not in self.color_palette:
            self.color_palette[track_id] = tuple(random.randint(50, 200) for _ in range(3))
        return self.color_palette[track_id]

    def _detect(self, frame):
        result = self.model(frame, verbose=False)[0]
        detections = []
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            if cls_id in self.target_ids:
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_id))
        return detections

    def _draw_track(self, frame, track, frame_width):
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())
        color = self._get_color(track_id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cls_id = track.get_det_class()
        cls_name = self.model.names.get(cls_id, "Object")
        text = f"{cls_name} ID:{track_id}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        if x1 + text_size[0] + 5 > frame_width:
            bg_x1, bg_x2 = frame_width - text_size[0] - 5, frame_width
        else:
            bg_x1, bg_x2 = x1, x1 + text_size[0] + 5
        bg_y1, bg_y2 = max(0, y1 - text_size[1] - 10), y1 - 10

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (255, 255, 255), -1)
        cv2.putText(frame, text, (bg_x1, y1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def process_frame(self, frame, frame_width):
        detections = self._detect(frame)
        tracks = self.tracker.update_tracks(detections, frame=frame)
        for track in tracks:
            if track.is_confirmed():
                self._draw_track(frame, track, frame_width)
        return frame

    def process_video(self, video_path, output_dir):
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "object_track_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        frame_cnt = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame = self.process_frame(frame, frame_width)
                out.write(frame)
                frame_cnt += 1
                if frame_cnt % 400 == 0:
                    print(f"Processed -{frame_cnt}- frames")
        except KeyboardInterrupt as e:
            print(f"Error: {e}")
        finally:
            cap.release()
            out.release()
            print(f'Video saved to {output_path}')
            print(f'Total frames = {frame_cnt}')

    def reset(self):
        self.color_palette = {}
        self.tracker = DeepSort(max_age=self.tracker.max_age)