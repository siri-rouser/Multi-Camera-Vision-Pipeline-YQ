import time
import heapq
import pybase64
from common import default_arg_parser, register_stop_handler
from visionapi_yq.messages_pb2 import SaeMessage
from visionlib.pipeline.publisher import RedisPublisher
from visionlib.saedump import DumpMeta, Event, message_splitter


def set_frame_timestamp_to_now(proto_bytes: bytes) -> bytes:
    proto = SaeMessage()
    proto.ParseFromString(proto_bytes)
    proto.frame.timestamp_utc_ms = time.time_ns() // 1_000_000
    return proto.SerializeToString()


class StreamEvent:
    def __init__(self, event: Event, stream_start_time: float, file_iter):
        self.event = event
        self.stream_start_time = stream_start_time
        self.file_iter = file_iter

    def __lt__(self, other):
        return (self.event.meta.record_time - self.stream_start_time) < \
               (other.event.meta.record_time - other.stream_start_time)


def stream_events_generator(file_path):
    with open(file_path, 'r') as file:
        messages = message_splitter(file)
        dump_meta = DumpMeta.model_validate_json(next(messages))
        for msg in messages:
            event = Event.model_validate_json(msg)
            yield event, dump_meta.start_time


if __name__ == '__main__':

    arg_parser = default_arg_parser()
    arg_parser.add_argument('dumpfiles', nargs='+')
    arg_parser.add_argument('-t', '--adjust-timestamps', action='store_true',
                            help='Adjust message timestamps to playback time')
    args = arg_parser.parse_args()

    stop_event = register_stop_handler()

    publisher = RedisPublisher(args.redis_host, args.redis_port)

    # Prepare initial heap
    heap = []
    streams_iterators = []

    for dumpfile in args.dumpfiles:
        file_iter = stream_events_generator(dumpfile)
        streams_iterators.append(file_iter)
        try:
            event, start_time = next(file_iter)
            heapq.heappush(heap, StreamEvent(event, start_time, file_iter))
        except StopIteration:
            continue

    playback_start = time.time()

    with publisher:
        while heap and not stop_event.is_set():
            next_stream_event = heapq.heappop(heap)

            event_time_offset = (next_stream_event.event.meta.record_time - next_stream_event.stream_start_time)
            playback_time_offset = time.time() - playback_start

            if playback_time_offset < event_time_offset:
                time.sleep(event_time_offset - playback_time_offset)

            proto_bytes = pybase64.standard_b64decode(next_stream_event.event.data_b64)

            if args.adjust_timestamps:
                proto_bytes = set_frame_timestamp_to_now(proto_bytes)

            publisher(next_stream_event.event.meta.source_stream, proto_bytes)

            # Fetch the next event from this stream
            try:
                event, _ = next(next_stream_event.file_iter)
                heapq.heappush(heap, StreamEvent(event, next_stream_event.stream_start_time, next_stream_event.file_iter))
            except StopIteration:
                continue