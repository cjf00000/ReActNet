clear
mkdir log
#python3 ./multiproc.py --nproc_per_node 8 ./train.py --data=/home/LargeData/Large/ImageNet --batch_size=64 --learning_rate=1e-3 --epochs=256 --weight_decay=1e-5 #| tee -a log/training.txt
#python3 train.py --batch_size=128 --dataset=cifar10 --learning_rate 0.03 --save=cifar10_fp18_c64 --data=~/data
#python3 train.py --batch_size=128 --dataset=cifar100 --learning_rate 0.1 --save=cifar100_fp18_c64_sgd_wd5e-4_la1lw1_w.1x --weight_decay=5e-4 --arch resnet18 --channel 64 --qa=l1 --qw=l1 --data=~/data --epochs 256
python3 train.py --resume fp.pth.tar --batch_size=128 --dataset=cifar100 --learning_rate 0.1 --save=cifar100_fp18_c64_sgd_wd5e-4_la2lw2_init --weight_decay=5e-4 --arch resnet18 --channel 64 --qa=l2 --qw=l2 --data=~/data --epochs 256
#python3 train.py --batch_size=128 --dataset=cifar100 --learning_rate 0.1 --save=cifar100_fp18_c192_sgd_wd5e-4 --weight_decay=5e-4 --arch resnet18 --channel 192 --qa=fp --qw=fp --data=~/data --epochs 256
#python3 ./multiproc.py --nproc_per_node 8 ./train.py --data=/home/LargeData/Large/ImageNet --dataset=imagenet --batch_size=64 --learning_rate=0.512 --epochs=90 --weight_decay=1e-5 --arch resnet18 --qa=b --qw=b --teacher resnet34 --save=imagenet_r18_c64_sgd_wd1e-5_distill_qaqw