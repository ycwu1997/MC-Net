nvidia-smi
python ./code/train_mcnet_2d.py  --model mcnet2d_v2 --labelnum 14 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_2d.py  --model mcnet2d_v2 --labelnum 7 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_2d.py  --model mcnet2d_v1 --labelnum 14 --gpu 0 --temperature 0.1 && \
python ./code/train_mcnet_2d.py  --model mcnet2d_v1 --labelnum 7 --gpu 0 --temperature 0.1 && \

python ./code/test_2d.py --exp MCNet --model mcnet2d_v2 --labelnum 7 --gpu 0 && \
python ./code/test_2d.py --exp MCNet --model mcnet2d_v2 --labelnum 14 --gpu 0 && \
python ./code/test_2d.py --exp MCNet --model mcnet2d_v1 --labelnum 7 --gpu 0 && \
python ./code/test_2d.py --exp MCNet --model mcnet2d_v1 --labelnum 14 --gpu 0
