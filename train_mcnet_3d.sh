nvidia-smi
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v2 --labelnum 16 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v2 --labelnum 8 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v1 --labelnum 16 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_3d.py --dataset_name LA --model mcnet3d_v1 --labelnum 8 --gpu 0 --temperature 0.1 && \

python ./code/test_3d.py --dataset_name LA --model mcnet3d_v2 --exp MCNet --labelnum 16 --gpu 0 && \
python ./code/test_3d.py --dataset_name LA --model mcnet3d_v2 --exp MCNet --labelnum 8 --gpu 0 && \
python ./code/test_3d.py --dataset_name LA --model mcnet3d_v1 --exp MCNet --labelnum 16 --gpu 0 && \
python ./code/test_3d.py --dataset_name LA --model mcnet3d_v1 --exp MCNet --labelnum 8 --gpu 0 && \

python ./code/train_mcnet_3d.py --dataset_name Pancreas_CT --model mcnet3d_v2 --labelnum 12 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_3d.py --dataset_name Pancreas_CT --model mcnet3d_v2 --labelnum 6 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_3d.py --dataset_name Pancreas_CT --model mcnet3d_v1 --labelnum 12 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_3d.py --dataset_name Pancreas_CT --model mcnet3d_v1 --labelnum 6 --gpu 0 --temperature 0.1 && \

python ./code/test_3d.py --dataset_name Pancreas_CT --model mcnet3d_v2 --exp MCNet --labelnum 12 --gpu 0 && \
python ./code/test_3d.py --dataset_name Pancreas_CT --model mcnet3d_v2 --exp MCNet --labelnum 6 --gpu 0 && \
python ./code/test_3d.py --dataset_name Pancreas_CT --model mcnet3d_v1 --exp MCNet --labelnum 12 --gpu 0 && \
python ./code/test_3d.py --dataset_name Pancreas_CT --model mcnet3d_v1 --exp MCNet --labelnum 6 --gpu 0
