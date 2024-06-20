import os
import sys
import tensorflow as tf
import pyarrow.parquet as pq

def disk_size(file):
    return os.path.getsize(file)

def tf_memory(tf_file):
    dataset = tf.data.TFRecordDataset(tf_file)
    return sys.getsizeof([record.numpy() for record in dataset])

def pq_memory(pq_file):
    return pq.read_table(pq_file).nbytes

DIR = os.path.expanduser("~") + "/datasets/"
in_dir  = DIR + "berkeley_autolab_ur5/"
out_dir = DIR + "parquet/"
prefix  = "berkeley_autolab_ur5-train.tfrecord-"
suffix  = ".parquet"

MB = 1024 * 1024
tf_file = in_dir + prefix + "00000-of-00412"
pq_file = out_dir + prefix + "00000.parquet"

print(f"TFRecord size on disk: {disk_size(tf_file) / MB:.2f} MB")
print(f"Parquet  size on disk: {disk_size(pq_file) / MB:.2f} MB")
print(f"TFRecord size in memory: {tf_memory(tf_file) / MB:.2f} MB")
print(f"Parquet  size in memory: {pq_memory(pq_file) / MB:.2f} MB")