export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export TRANSFORMERS_CACHE=/ceph/jbrinkma/cache/transformers
export HF_DATASETS_CACHE=/ceph/jbrinkma/cache/datasets

python3 run.py