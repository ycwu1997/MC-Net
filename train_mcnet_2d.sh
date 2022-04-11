nvidia-smi
python ./mcnet/train_mcnet_2d.py  --model mcnet2d_v2 --labelnum 14 --gpu 0 --temperature 0.1 && \
python ./mcnet/train_mcnet_2d.py  --model mcnet2d_v2 --labelnum 7 --gpu 0 --temperature 0.1 && \
python ./mcnet/train_mcnet_2d.py  --model mcnet2d_v1 --labelnum 14 --gpu 0 --temperature 0.1 && \
python ./mcnet/train_mcnet_2d.py  --model mcnet2d_v1 --labelnum 7 --gpu 0 --temperature 0.1 && \

python ./mcnet/train_baseline_2d.py --model unet --labelnum 7 --gpu 0  && \
python ./mcnet/train_baseline_2d.py --model unet --labelnum 14 --gpu 0  && \
python ./mcnet/train_baseline_2d.py --model unet --labelnum 70 --gpu 0  && \

python ./mcnet/test_2d.py --exp MCNet --model mcnet2d_v2 --labelnum 7 --gpu 0 && \
python ./mcnet/test_2d.py --exp MCNet --model mcnet2d_v2 --labelnum 14 --gpu 0 && \
python ./mcnet/test_2d.py --exp MCNet --model mcnet2d_v1 --labelnum 7 --gpu 0 && \
python ./mcnet/test_2d.py --exp MCNet --model mcnet2d_v1 --labelnum 14 --gpu 0 && \

python ./mcnet/test_2d.py --exp Baseline_2d --model unet --labelnum 7 --gpu 0 && \
python ./mcnet/test_2d.py --exp Baseline_2d --model unet --labelnum 14 --gpu 0 && \
python ./mcnet/test_2d.py --exp Baseline_2d --model unet --labelnum 70 --gpu 0