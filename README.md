# pytorch-classifier

## How to run
```bash
UDA_VISIBLE_DEVICES=0 python train.py -dataset hmdb51 -batch_size 25 -step_size 8 -num_epochs 30 -tr_pth /data2/yunfeng/dataset/hmdb51/obj_box_3_per_video/train -val_pth /data2/yunfeng/dataset/hmdb51/obj_box_3_per_video/val -lr 0.01 2>&1 | tee hmdb51_train_obj_box_2_per_video_lr_0.01.log
```
