#!/usr/bin/sh
python3 main.py \
    --resolution-high 512 \
    --resolution-wide 512 \
    --nlatent 9 \
    --dataset-train filelist \
    --filename-train ./data/data.txt \
    --dataset-test filelist \
    --filename-test ./data/testdata.txt \
    --loader-train h5py \
    --nthreads 32 \
    --nepochs 500 \
    --nchannels 3 \
    --nechannels 6 \
    --ngchannels 1 \
    --ndf 8 \
    --nef 8 \
    --learning-rate 1e-2 \
    --adam-beta1 0.5 \
    --adam-beta2 0.999 \
    --momentum 0.00 \
    --weight-decay 0.00 \
    --batch-size 64 \
    --out-images 32 \
    --wkld 0.01 \
    --out-steps 20 \
    --save-steps 10 \
    --optim-method "RMSprop" \
    # --resume '{"netE": "/zdata/icurtis/deep-learning/microstructure-recognition/vishnu-collab/micro-vae-vishnu/results/2018-08-06_17-55-01/Save", "netG": "/zdata/icurtis/deep-learning/microstructure-recognition/vishnu-collab/micro-vae-vishnu/results/2018-08-06_17-55-01/Save"}' \