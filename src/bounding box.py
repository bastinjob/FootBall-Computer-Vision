from tqdm import tqdm
import supervision as sv
from get_model import get_model

source_video_path = "/path_to_input_video/xxxx.mp4"
target_video_path = "/path_to_output_video/yyyy.mp4"

DETECTION_MODEL = get_model("your_model_id")

box_annotator = sv.BoxAnnotator(
    color = sv.ColorPalette.from_hex(["#FF8C00","#00BFFF","#FF1493","#FFD700"]),
    thickness = 2
)
label_annotator = sv.LabelAnnotator(
     color = sv.ColorPalette.from_hex(["#FF8C00","#00BFFF","#FF1493","#FFD700"]),
     text_color = sv.Color.from_hex("#000000")
)

video_info = sv.VideoInfo.from_video_path(source_video_path)
video_sink = sv.VideoSink(target_video_path, video_info)

frame_generator = sv.get_video_frames_generator(source_video_path)
with video_sink:
  for frame in tqdm(frame_generator, total=video_info.total_frames):
    #frame  = next(frame_generator)

    result = DETECTION_MODEL.infer(frame, confidence = 0.3)[0]
    detections = sv.Detections.from_inference(result)

    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections["class_name"], detections.confidence)
    ]

    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections,
        labels=labels
    )
    video_sink.write_frame(annotated_frame)