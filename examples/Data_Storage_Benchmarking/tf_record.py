import tensorflow as tf
import time
import os

read_files = [
    '/home/joshzhang/datasets/berkeley_autolab_ur5/0.1.0/berkeley_autolab_ur5-test.tfrecord-00042-of-00050',
    '/home/joshzhang/datasets/berkeley_autolab_ur5/0.1.0/berkeley_autolab_ur5-train.tfrecord-00069-of-00412'
]
write_file = '/home/joshzhang/datasets/berkeley_autolab_ur5/0.1.0/berkeley_autolab_ur5-train_write_test.tfrecord'

def measure_read_speed(files):
    start_time = time.time()
    total_bytes = 0

    for file_path in files:
        for record in tf.data.TFRecordDataset(file_path):
            total_bytes += len(record.numpy())

    end_time = time.time()
    elapsed_time = end_time - start_time
    read_speed = total_bytes / elapsed_time / (1024 * 1024)  # MB/s

    print(f"Read {total_bytes} bytes in {elapsed_time:.2f} seconds")
    print(f"Read Speed: {read_speed:.2f} MB/s")

def measure_write_speed(files, output_file):
    start_time = time.time()
    total_bytes = 0

    with tf.io.TFRecordWriter(output_file) as writer:
        for file_path in files:
            for record in tf.data.TFRecordDataset(file_path):
                writer.write(record.numpy())
                total_bytes += len(record.numpy())

    end_time = time.time()
    elapsed_time = end_time - start_time
    write_speed = total_bytes / elapsed_time / (1024 * 1024)  # MB/s

    print(f"Wrote {total_bytes} bytes in {elapsed_time:.2f} seconds")
    print(f"Write Speed: {write_speed:.2f} MB/s")

print("Measuring Read Speed...")
measure_read_speed(read_files)
print("Measuring Write Speed...")
measure_write_speed(read_files, write_file)
os.remove(write_file)