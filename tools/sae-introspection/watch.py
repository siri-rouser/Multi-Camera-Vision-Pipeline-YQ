import sys
import os
import time

import cv2
import numpy as np
import redis
from common import choose_stream, default_arg_parser, register_stop_handler
from visionapi_yq.messages_pb2 import Detection, SaeMessage
from visionlib.pipeline.consumer import RedisConsumer
from visionlib.pipeline.tools import get_raw_frame_data

ANNOTATION_COLOR = (0, 0, 255)
DEFAULT_WINDOW_SIZE = (1280, 720)

previous_frame_timestamp = 0
args = None

def average(lst):
    return sum(lst) / len(lst) if lst else 0

def isWindowVisible(window_name):
    try:
        windowVisibleProp = int(cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE))
        return windowVisibleProp == 1
    except:
        return False
    
def get_image(sae_msg: SaeMessage):
    if args.image_file is not None:
        image = cv2.imread(args.image_file)
        if image is None:
            raise ValueError(f'Could not read image from file {args.image_file}')
        return image
    else:
        frame = get_raw_frame_data(sae_msg.frame)
        if frame is not None:
            return frame
        else:
            # If no frame is available, return a grey image as a last resort
            return np.ones((sae_msg.frame.shape.height, sae_msg.frame.shape.width, 3), dtype=np.uint8) * 127


def annotate(image, detection: Detection):
    bbox_x1 = int(detection.bounding_box.min_x * image.shape[1])
    bbox_y1 = int(detection.bounding_box.min_y * image.shape[0])
    bbox_x2 = int(detection.bounding_box.max_x * image.shape[1])
    bbox_y2 = int(detection.bounding_box.max_y * image.shape[0])

    class_id = detection.class_id
    conf = detection.confidence

    label = f'{class_id} - {round(conf,2)}'

    if detection.object_id is not None:
        object_id = detection.object_id
        label = f'ID {object_id} - {class_id} - {round(conf,2)}'

    line_width = max(round(sum(image.shape) / 2 * 0.002), 2)

    cv2.rectangle(image, (bbox_x1, bbox_y1), (bbox_x2, bbox_y2), color=ANNOTATION_COLOR, thickness=line_width, lineType=cv2.LINE_AA)
    cv2.putText(image, label, (bbox_x1, bbox_y1 - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=ANNOTATION_COLOR, thickness=round(line_width/3), fontScale=line_width/4, lineType=cv2.LINE_AA)

def showImage(stream_id, image):
    if not isWindowVisible(window_name=stream_id):
        cv2.namedWindow(stream_id, cv2.WINDOW_NORMAL + cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(stream_id, *DEFAULT_WINDOW_SIZE)
        
    cv2.imshow(stream_id, image)
    if cv2.waitKey(1) == ord('q'):
        stop_event.set()
        cv2.destroyAllWindows()

def handle_sae_message(sae_message_bytes, stream_key, time_dict):
    global previous_frame_timestamp, args

    sae_msg = SaeMessage()
    sae_msg.ParseFromString(sae_message_bytes)

    frametime = sae_msg.frame.timestamp_utc_ms - previous_frame_timestamp
    previous_frame_timestamp = sae_msg.frame.timestamp_utc_ms
    log_line = f'E2E-Delay: {round(time.time() * 1000 - sae_msg.frame.timestamp_utc_ms): >8} ms, Display Frametime: {frametime: >5} ms'
    if sae_msg.HasField('metrics'):
        log_line_detail = f', Detection: {sae_msg.metrics.detection_inference_time_us: >7} us, Feature_extraction: {sae_msg.metrics.feature_extraction_time_us: >7} us, Tracking: {sae_msg.metrics.tracking_inference_time_us: >7} us, Merge: {sae_msg.metrics.merge_inference_time_us: >7} us'
        print(log_line_detail, file=sys.stderr)
        time_dict['detection_time'].append(sae_msg.metrics.detection_inference_time_us)
        time_dict['feature_extraction_time'].append(sae_msg.metrics.feature_extraction_time_us)
        time_dict['tracking_time'].append(sae_msg.metrics.tracking_inference_time_us)
        time_dict['merge_inference_time'].append(sae_msg.metrics.merge_inference_time_us)
    print(log_line, file=sys.stderr)
    avg_log_line = f'average_detection_time:{average(time_dict["detection_time"]): >7} us, average_feature_extraction_time:{average(time_dict["feature_extraction_time"]): >7} us, average_tracking_time:{average(time_dict["tracking_time"]): >7} us, average_merge_inference_time:{average(time_dict["merge_inference_time"]): >7} us'
    print(avg_log_line, file=sys.stderr)
    image = get_image(sae_msg)

    # Annotate the timestamp at the top-left corner of the frame
    timestamp_text = f'Timestamp: {sae_msg.frame.timestamp_utc_ms} ms, frame_id: {sae_msg.frame.frame_id}'
    font_scale = 2
    font_thickness = 1
    text_color = (255, 255, 255)  # White
    text_background_color = (0, 0, 0)  # Black
    text_size, _ = cv2.getTextSize(timestamp_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    text_x, text_y = 10, 10 + text_size[1]
    cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5), (text_x + text_size[0] + 5, text_y + 5), text_background_color, -1)
    cv2.putText(image, timestamp_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thickness, cv2.LINE_AA)

    for detection in sae_msg.detections:
        annotate(image, detection)
    
    if args.stdout:
        sys.stdout.buffer.write(image)
    
    showImage(stream_key, image)

    # For checking time stamp
    # print('sae_msg timestamp:', sae_msg.frame.frame_id)

    return image, time_dict
    


if __name__ == '__main__':

    arg_parser = default_arg_parser()
    arg_parser.add_argument('-s', '--stream', type=str)
    arg_parser.add_argument('-i', '--image-file', type=str, default=None)
    arg_parser.add_argument('-o', '--stdout', action='store_true', help='Output annotated raw frames to stdout (e.g. to pipe into ffmpeg)')
    args = arg_parser.parse_args()

    if args.stdout and sys.stdout.isatty():
        print('Stdout is the terminal. Ignoring "stdout" option. Please redirect (e.g. into ffmpeg)', file=sys.stderr)
        args.stdout = False

    STREAM_KEY = args.stream
    REDIS_HOST = args.redis_host
    REDIS_PORT = args.redis_port

    if STREAM_KEY is None:
        redis_client = redis.Redis(REDIS_HOST, REDIS_PORT)
        STREAM_KEY = choose_stream(redis_client)
    
    stop_event = register_stop_handler()

    consume = RedisConsumer(REDIS_HOST, REDIS_PORT, [STREAM_KEY], block=200)

    video_writer = cv2.VideoWriter('record.mp4', cv2.VideoWriter_fourcc(*'mp4v') , 10.0, (2560, 1440))
    time_dict = {}
    time_dict['detection_time'] = []
    time_dict['feature_extraction_time'] = []
    time_dict['tracking_time'] = []
    time_dict['merge_inference_time'] = []

    with consume:
        for stream_key, proto_data in consume():
            if stop_event.is_set():
                break

            if stream_key is None:
                continue
            
            image,time_dict = handle_sae_message(proto_data, stream_key,time_dict)

            video_writer.write(image)