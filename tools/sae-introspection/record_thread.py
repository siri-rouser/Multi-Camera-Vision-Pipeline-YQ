#!/usr/bin/env python3
import os
import time
import threading
import queue
from typing import TextIO

import cv2
import pybase64
import redis
from turbojpeg import TurboJPEG

from visionapi_yq.messages_pb2 import SaeMessage
from visionlib.pipeline.consumer import RedisConsumer
from visionlib.pipeline.tools import get_raw_frame_data
from visionlib.saedump import MESSAGE_SEPARATOR, DumpMeta, Event, EventMeta

from common import choose_streams, default_arg_parser, register_stop_handler

output_file_handle: TextIO | None = None
written_count = 0

# ------------- Helpers -------------
def write_meta(file: TextIO, start_time: float, stream_keys: list[str]):
    meta = DumpMeta(
        start_time=start_time,
        recorded_streams=stream_keys
    )
    file.write(meta.model_dump_json())
    file.write(MESSAGE_SEPARATOR)

def write_event(
    file: TextIO,
    stream_key: str,
    proto_data: bytes,
    *,
    is_remove_frame: bool,
    scale_width: int,
    scale_quality: int,
    jpeg: TurboJPEG
):
    """Optionally strip frame or downscale+re-encode, then write Event JSON to file."""
    bytes_to_write = proto_data

    if is_remove_frame:
        bytes_to_write = remove_frame(proto_data)
    elif scale_width > 0:
        # Only do resize if we didn't remove the frame entirely
        bytes_to_write = resize_frame(proto_data, scale_width, scale_quality, jpeg)

    event = Event(
        meta=EventMeta(
            record_time=time.time(),
            source_stream=stream_key
        ),
        data_b64=pybase64.standard_b64encode(bytes_to_write)
    )
    file.write(event.model_dump_json())
    file.write(MESSAGE_SEPARATOR)

def remove_frame(proto_data: bytes) -> bytes:
    msg = SaeMessage()
    msg.ParseFromString(proto_data)
    msg.frame.ClearField('frame_data')
    msg.frame.ClearField('frame_data_jpeg')
    return msg.SerializeToString()

def resize_frame(proto_data: bytes, scale_width: int, quality: int, jpeg: TurboJPEG) -> bytes:
    msg = SaeMessage()
    msg.ParseFromString(proto_data)

    # Get raw frame regardless of whether it's raw or jpeg inside the proto
    frame = get_raw_frame_data(msg.frame)
    if frame is None:
        # If no frame available, just return original bytes (nothing to resize)
        return proto_data

    original_width = frame.shape[1]
    if original_width <= 0 or scale_width <= 0:
        return proto_data

    scale_factor = scale_width / float(original_width)
    if scale_factor <= 0:
        return proto_data

    # Resize and re-encode JPEG into the message
    resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    msg.frame.ClearField('frame_data')  # ensure we only keep JPEG to keep message small
    msg.frame.frame_data_jpeg = jpeg.encode(resized, quality)
    return msg.SerializeToString()

# ------------- Main -------------
if __name__ == '__main__':
    arg_parser = default_arg_parser()
    arg_parser.add_argument("-s", "--streams", type=str, nargs="*", metavar="STREAM")
    arg_parser.add_argument('-t', '--time-limit', type=int, help='Stop after TIME_LIMIT seconds (default 650)', default=650)
    arg_parser.add_argument('-r', '--remove-frame', action='store_true', help='Remove frame data from messages')
    arg_parser.add_argument('-d', '--downscale-frames', default=0, type=int, help='Downscale frames to given width (preserve aspect ratio)')
    arg_parser.add_argument('-q', '--downscale-jpeg-quality', default=85, type=int, help='JPEG quality for downscaling (0-100)')
    args = arg_parser.parse_args()

    STREAM_KEYS = args.streams
    REDIS_HOST = args.redis_host
    REDIS_PORT = args.redis_port

    if STREAM_KEYS is None:
        redis_client = redis.Redis(REDIS_HOST, REDIS_PORT)
        STREAM_KEYS = choose_streams(redis_client)

    # Output dirs
    saedump_dir = '/home/yuqiang/yl4300/Multi-Camera-Vision-Pipeline-YQ/tools/sae-introspection/record_saedump'
    os.makedirs(saedump_dir, exist_ok=True)

    output_file = os.path.join(saedump_dir, f'{STREAM_KEYS[0]}.saedump')

    print(f"Will record stream(s): {STREAM_KEYS}")
    print(f"Output: {output_file}")
    print(f"Time limit (from first message): {args.time_limit}s")

    stop_event = register_stop_handler()
    consume = RedisConsumer(REDIS_HOST, REDIS_PORT, STREAM_KEYS, block=200)

    # --- Producer/Consumer plumbing ---
    WRITE_QUEUE_MAX = 30  # ~15 seconds of buffer at 15 FPS for one stream; increase if multiple streams/heavier transforms
    BATCH_SIZE = 1             # number of events per flush to reduce syscall overhead
    FSYNC_EVERY = 0            # set N>0 to fsync every N batches; 0 disables explicit fsync

    write_q: "queue.Queue[tuple[str, bytes] | None]" = queue.Queue(maxsize=WRITE_QUEUE_MAX)

    started = False
    start_time = None
    output_file_handle: TextIO | None = None

    # Counters for sanity checks at shutdown
    enqueued_count = 0
    written_count = 0

    stop_writer = threading.Event()
    writer_started_evt = threading.Event()
    
    def writer_thread_fn():
        global output_file_handle, written_count  # these are module-level

        local_jpeg = TurboJPEG()  # thread-local encoder
        batch: list[tuple[str, bytes]] = []
        batches_since_fsync = 0

        def flush_batch():
            nonlocal batch, batches_since_fsync          # enclosing function vars
            global written_count, output_file_handle     # module-level vars
            if not batch:
                return
            for (stream_key, proto_data) in batch:
                # Guard: output_file_handle can be None if something failed early
                if output_file_handle is None:
                    continue
                write_event(
                    output_file_handle,
                    stream_key,
                    proto_data,
                    is_remove_frame=args.remove_frame,
                    scale_width=args.downscale_frames,
                    scale_quality=args.downscale_jpeg_quality,
                    jpeg=local_jpeg
                )
                written_count += 1
            batch = []  # or: batch.clear()
            batches_since_fsync += 1
            if FSYNC_EVERY and output_file_handle is not None and (batches_since_fsync % FSYNC_EVERY == 0):
                output_file_handle.flush()
                os.fsync(output_file_handle.fileno())


        try:
            writer_started_evt.set()
            while not stop_writer.is_set():
                try:
                    item = write_q.get(timeout=0.2)
                except queue.Empty:
                    flush_batch()
                    continue

                if item is None:
                    # Poison pill for clean shutdown
                    flush_batch()
                    break

                stream_key, proto_data = item
                batch.append((stream_key, proto_data))
                if len(batch) >= BATCH_SIZE:
                    flush_batch()
                write_q.task_done()

            # Final flush
            flush_batch()
        finally:
            if output_file_handle:
                output_file_handle.flush()

    # --- Run ---
    try:
        with consume:
            for stream_key, proto_data in consume():
                if stop_event.is_set():
                    break
                if stream_key is None or proto_data is None:
                    continue

                if not started:
                    started = True
                    start_time = time.time()
                    # Create file on first data to avoid empty dumps
                    # Large buffer to reduce syscalls
                    output_file_handle = open(output_file, "x", buffering=1024 * 1024)
                    write_meta(output_file_handle, start_time, STREAM_KEYS)
                    print("First message received — recording started.")

                    # Start writer thread once we have a file
                    t = threading.Thread(target=writer_thread_fn, daemon=True)
                    t.start()
                    writer_started_evt.wait()

                # Enqueue quickly; applies back-pressure if writer lags
                write_q.put((stream_key, proto_data))
                enqueued_count += 1

                if time.time() - start_time > args.time_limit:
                    print(f"Reached configured time limit of {args.time_limit}s")
                    break

    finally:
        # Stop writer and drain
        stop_writer.set()
        try:
            write_q.put_nowait(None)  # poison pill
        except queue.Full:
            # If queue is full, block until space is available
            write_q.put(None)
            print("Warning: write queue was full during shutdown; blocked briefly.")

        if output_file_handle:
            output_file_handle.close()

        print(f"Enqueued: {enqueued_count}, Written: {written_count}")
