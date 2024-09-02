# ask for sudo access
sudo echo "Use sudo access for clearning cache"

rm *.csv

# Define a list of batch sizes to iterate through
batch_sizes=(1 8)
# batch_sizes=(1 2)

num_batches=1000

# Iterate through each batch size
for batch_size in "${batch_sizes[@]}"
do
    echo "Running benchmarks with batch size: $batch_size"
    
    # python3 benchmarks/openx.py --dataset_names nyu_door_opening_surprising_effectiveness --num_batches $num_batches --batch_size $batch_size
    python3 benchmarks/openx.py --dataset_names berkeley_autolab_ur5 --num_batches $num_batches --batch_size $batch_size
    # python3 benchmarks/openx.py --dataset_names berkeley_cable_routing --num_batches $num_batches --batch_size $batch_size
    # python3 benchmarks/openx.py --dataset_names bridge --num_batches $num_batches --batch_size $batch_size
done