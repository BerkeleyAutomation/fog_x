import fog_x
import numpy as np
import time 

path = "/tmp/output.vla"


# import av

# def remux_mkv(input_filename, output_filename):
#     # Open the input file using PyAV
#     input_container = av.open(input_filename, format = "matroska")

#     # Create an output container for the new file
#     output_container = av.open(output_filename, mode='w', format='matroska')

#     # Loop through all streams in the input file and add them to the output file
#     for stream in input_container.streams:
#         output_container.add_stream(stream.codec_context.codec.name)
#         print(stream.codec_context.codec.name)

#     # Read packets from the input file and write them to the output file
#     for packet in input_container.demux():
#         if packet.dts is None:
#             print("Skipping packet with no dts")
#             continue
#         stream = output_container.streams[packet.stream.index]
#         print(packet.stream.metadata, packet)
#         packet.stream = stream
#         output_container.mux(packet)

#     # Close both containers
#     output_container.close()
#     input_container.close()

# input_filename = "/home/kych/datasets/rtx/mkv_convert/output_0.mkv"#"/tmp/output.vla"
# input_filename = "/tmp/output.vla"
# output_filename = "/tmp/remuxed.mkv"

# remux_mkv(input_filename, output_filename)

# exit(0)

# remove the existing file
import os
os.system(f"rm -rf {path}")
os.system(f"rm -rf /tmp/*.cache")

# 🦊 Data collection: 
# create a new trajectory
traj = fog_x.Trajectory(
    path = path
)

# collect step data for the episode
for i in range(100):
    time.sleep(0.001)
    traj.add(feature = "arm_view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "gripper_pose", data = np.ones((4, 4), dtype=np.float32))
    traj.add(feature = "view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "wrist_view", data = np.ones((640, 480, 3), dtype=np.uint8))
    traj.add(feature = "joint_angles", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "joint_velocities", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "joint_torques", data = np.ones((7,), dtype=np.float32))
    traj.add(feature = "ee_force", data = np.ones((6,), dtype=np.float32))
    traj.add(feature = "ee_velocity", data = np.ones((6,), dtype=np.float32))
    traj.add(feature = "ee_pose", data = np.ones((4, 4), dtype=np.float32))

traj.close()


traj = fog_x.Trajectory(
    path = path
)