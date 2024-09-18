

# # bridge dataset
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name bridge --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name bridge --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[0:] --max_workers 16  --lossless

# berkeley_cable_routing dataset
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_cable_routing --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_cable_routing --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[0:] --max_workers 16  --lossless
# python examples/fixing_failed_conversions.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_cable_routing --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 16

# nyu_door_opening_surprising_effectiveness dataset
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name nyu_door_opening_surprising_effectiveness --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name nyu_door_opening_surprising_effectiveness --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[0:] --max_workers 16  --lossless
# python examples/fixing_failed_conversions.py --data_dir /home/kych/datasets/rtx --dataset_name nyu_door_opening_surprising_effectiveness --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 16

# bridge dataset
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name bridge --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[6000:] --max_workers 16
# pkill -f examples
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name bridge --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[0:] --max_workers 16 --lossless
python examples/fixing_failed_conversions.py --data_dir /home/kych/datasets/rtx --dataset_name bridge --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 8
pkill -f examples

# berkeley_autolab_ur5 dataset
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[400:] --max_workers 16
# pkill -f examples
python examples/fixing_failed_conversions.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[0:] --max_workers 8
pkill -f examples


# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[200:400] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[400:600] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[600:800] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/vla --version 0.1.0 --split train[800:] --max_workers 16 

# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[0:] --max_workers 16  --lossless
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[200:400] --max_workers 16 --lossless
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[400:600] --max_workers 16 --lossless
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[600:800] --max_workers 16 --lossless
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name berkeley_autolab_ur5 --destination_dir /mnt/data/fog_x/ffv1 --version 0.1.0 --split train[800:] --max_workers 16 --lossless


# fractal20220817_data
# rm -rf /home/kych/datasets/fractal20220817_data/vla
# rm -rf /home/kych/datasets/fractal20220817_data/ffv1
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name fractal20220817_data --destination_dir /home/kych/datasets/fractal20220817_data/vla --version 0.1.0 --split train[34000:] --max_workers 16
# python examples/openx_loader.py --data_dir /home/kych/datasets/rtx --dataset_name fractal20220817_data --destination_dir /home/kych/datasets/fractal20220817_data/ffv1 --version 0.1.0 --split train[0:] --max_workers 8 --lossless

