import pyturtlmap as tm
from utils.h5_log_reader import H5LogReader
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("config_file")
parser.add_argument("h5_file")
args = parser.parse_args()

barometer_key = "BlueROV_pressure2_fluid"
dvl_key = "dvl_data"
imu_key = "vectornav_IMU"
stereo_key = "zed"

data_streamer = H5LogReader(args.h5_file, imu_key, dvl_key, barometer_key, stereo_key)
data_provider = tm.LiveFeedDataProvider()
pg_backend = tm.PoseGraphBackend(args.config_file, data_provider)

pg_backend.run()
data_provider.start()

events = data_streamer.events()
t0 = events[0].stamp
wall_t0 = time.time()

for event in events:
    # compute real-time delay
    elapsed_wall = time.time() - wall_t0
    elapsed_stamp = event.stamp - t0
    delay = elapsed_stamp - elapsed_wall
    if delay > 0:
        time.sleep(delay)

    if event.type == "IMU":
        m = data_streamer.imu()[event.index]
        data_provider.insert_imu(m)
    elif event.type == "DVL":
        m = data_streamer.dvl()[event.index]
        data_provider.insert_dvl(m)
    elif event.type == "PRESSURE":
        m = data_streamer.pressure()[event.index]
        data_provider.insert_baro(m)