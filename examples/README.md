Real steps:
1. run `gsutil -m cp -r gs://gresearch/robotics/nyu_door_opening_surprising_effectiveness/ /mnt/nvme0n1/xinyu/robodm/rlds/` to download RLDS
2. use examples/rlds_to_vla_batch_conversion.py to convert dataset to vla format
4. use examples/vla_to_hdf5_conversion.py to convert vla to hdf5 format
5. Dowload lerobot format from using
```bash
hf download \
      IPEC-COMMUNITY/nyu_door_opening_surprising_effectiveness_lerobot \
      --repo-type dataset \
      --local-dir $BASE_DIR/hf/nyu_door_opening_surprising_effectiveness
```