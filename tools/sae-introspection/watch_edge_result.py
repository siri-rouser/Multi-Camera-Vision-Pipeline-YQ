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

def isWindowVisible(window_name):
    try:
        windowVisibleProp = int(cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE))
        return windowVisibleProp == 1
    except:
        return False


def handle_sae_message(sae_message_bytes, stream_key):
    global previous_frame_timestamp, args

    if not sae_message_bytes:
        return  # Early exit if message is empty

    print(f'Received a message from feature_extractor')

    sae_msg = SaeMessage()
    sae_msg.ParseFromString(sae_message_bytes)

    camera_id = stream_key.split(':')[1]

    for track_id in sae_msg.trajectory.cameras[camera_id].tracklets:
        tracklet = sae_msg.trajectory.cameras[camera_id].tracklets[track_id]
        print(f'Track ID: {track_id}')
        print(f'with {len(tracklet.detections_info)} detections')
        print(f'The start frame is {tracklet.start_time}, the end frame is {tracklet.end_time}, the duration is {tracklet.end_time - tracklet.start_time}')
        print(f'shape of mean_feature is ({len(tracklet.mean_feature)},)')
        print('-------------------')
        

    # frametime = sae_msg.frame.timestamp_utc_ms - previous_frame_timestamp
    # previous_frame_timestamp = sae_msg.frame.timestamp_utc_ms

    # log_line = f'E2E-Delay: {round(time.time() * 1000 - sae_msg.frame.timestamp_utc_ms): >8} ms, Display Frametime: {frametime: >5} ms'
    # if sae_msg.HasField('metrics'):
    #     log_line += f', Detection: {sae_msg.metrics.detection_inference_time_us: >7} us, Tracking: {sae_msg.metrics.tracking_inference_time_us: >7} us'
    # print(log_line, file=sys.stderr)


if __name__ == '__main__':

    arg_parser = default_arg_parser()
    arg_parser.add_argument('-s', '--stream', type=str)
    arg_parser.add_argument('-i', '--image-file', type=str, default=None)
    arg_parser.add_argument('-o', '--stdout', action='store_true', help='Output annotated raw frames to stdout (e.g. to pipe into ffmpeg)')
    arg_parser.add_argument('-f', '--fixed-scale', type=float, 
                           help='Display with fixed scaling factor and high-quality scaling (2=double size, 1=original size, 0.75=75%% size , 0.5=half size, etc.)')

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

    print(f'Watching stream {STREAM_KEY} on {REDIS_HOST}:{REDIS_PORT}...')

    with consume:
        for stream_key, proto_data in consume():
            if stop_event.is_set():
                break

            if stream_key is None:
                continue
            
            handle_sae_message(proto_data, stream_key)