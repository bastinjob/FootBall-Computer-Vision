from tqdm import tqdm
import supervision as sv
from get_model import get_model

source_video_path = "/path_to_input_video/xxxx.mp4"
target_video_path = "/path_to_output_video/yyyy.mp4"

DETECTION_MODEL = get_model("your_model_id")

BALL_ID = 0



ellipse_annotator = sv.EllipseAnnotator(
    color = sv.ColorPalette.from_hex(["#00BFFF","#FF1493","#FFD700"]),
    thickness = 2
)

triangle_annotator = sv.TriangleAnnotator(
    color = sv.Color.from_hex("#FFD700"),
    base = 20,
    height = 17
)
#label_annotator = sv.LabelAnnotator(
#     color = sv.ColorPalette.from_hex("#FF8C00","#00BFFF","#FF1493","#FFD700"]),
#     text_color = sv.Color.from_hex("#000000")
#)

video_info = sv.VideoInfo.from_video_path(source_video_path)
video_sink = sv.VideoSink(target_video_path, video_info)

frame_generator = sv.get_video_frames_generator(source_video_path)
with video_sink:
  for frame in tqdm(frame_generator, total=video_info.total_frames):
    #frame  = next(frame_generator)

    result = DETECTION_MODEL.infer(frame, confidence = 0.3)[0]
    detections = sv.Detections.from_inference(result)
    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections.class_id = all_detections.class_id - 1
    #labels = [
     #   f"{class_name} {confidence:.2f}"
      #  for class_name, confidence
       # in zip(detections["class_name"], detections.confidence)
    #]

    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame, detections=all_detections
    )
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame, detections=ball_detections
    )
    video_sink.write_frame(annotated_frame)
