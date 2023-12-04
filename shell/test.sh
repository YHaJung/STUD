#!/bin/bash

# mkdir runs/seaships
# mkdir runs/seaships/res_figs
# python -m tools.train_net \
#     --config-file configs/seaships/stud_resnet.yaml \
#     --savefigdir './runs/seaships/res_figs' \
#     --visualize \
#     --num-gpus 1 \
#     --eval-only \
#     MODEL.WEIGHTS runs/seaships/model_final.pth \
# #     # OUTPUT_DIR ./runs/seaships \
#     # DATASETS.TEST \("MID_01_test_ood"\)

for num in 01 02 03 04 05 06 07; do
    python -m tools.train_net \
        --config-file configs/MID/stud_resnet_$num.yaml \
        --savefigdir ./runs/MID/$01/res_figs \
        --visualize \
        --num-gpus 1 \
        --eval-only \
        MODEL.WEIGHTS runs/seaships/model_final.pth
done

for num in 01 02 03 04 05 06 07 08 09 10 11 12; do
    python -m tools.train_net \
        --config-file configs/modd/stud_resnet_$num.yaml \
        --savefigdir ./runs/modd/$01/res_figs \
        --visualize \
        --num-gpus 1 \
        --eval-only \
        MODEL.WEIGHTS runs/seaships/model_final.pth
done

