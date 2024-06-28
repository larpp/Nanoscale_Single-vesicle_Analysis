# Command

## Dataset

```
dest/
  test/
    class1/
      img1.png
    class2/
      img2.png
  train/
    class1/
      img3.png
    class2/
      img4.png
  val/
    class1/
      img5.png
    class2/
      img6.png
```

## Training

The pre-trained models can be downloaded from the [official ConvNeXt Github](https://github.com/facebookresearch/ConvNeXt?tab=readme-ov-file).

```
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model convnext_small --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path dest \
--output_dir out \
--resume convnext_small_1k_224_ema.pth \
--nb_classes 2
```
