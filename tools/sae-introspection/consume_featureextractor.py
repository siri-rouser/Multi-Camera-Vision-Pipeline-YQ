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


def sae_msg_processor(proto_data, stream_key, outfile):

    sae_msg = SaeMessage()
    sae_msg.ParseFromString(proto_data)

    camera_id = stream_key.split(':')[1]

    for track_id in sae_msg.trajectory.cameras[camera_id].tracklets:
        tracklet = sae_msg.trajectory.cameras[camera_id].tracklets[track_id]

        for detection in tracklet.detections_info:
            detection: Detection
            frame_id = detection.frame_id
            track_id = detection.object_id
            x1 = detection.bounding_box.min_x * 3840
            y1 = detection.bounding_box.min_y * 2160
            x2 = detection.bounding_box.max_x * 3840
            y2 = detection.bounding_box.max_y * 2160
            line = f'{frame_id} {track_id} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n'

            # Write line to file
            outfile.write(line)
            outfile.flush()  # make sure it is written immediately

if __name__ == '__main__':

    arg_parser = default_arg_parser()
    arg_parser.add_argument('-s', '--stream', type=str)
    arg_parser.add_argument('-i', '--image-file', type=str, default=None)
    arg_parser.add_argument('-o', '--stdout', action='store_true', help='Output annotated raw frames to stdout (e.g. to pipe into ffmpeg)')
    arg_parser.add_argument('--out-txt', type=str, default='output.txt', help='Path to save txt results')
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

    with open(args.out_txt, "w") as outfile:   # append mode, will keep adding lines
        with consume:
            for stream_key, proto_data in consume():
                if stop_event.is_set():
                    break

                if stream_key is None:
                    continue
            
                sae_msg_processor(proto_data, stream_key, outfile)