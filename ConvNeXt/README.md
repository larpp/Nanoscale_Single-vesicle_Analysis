# Command


## Setup

You can check the environment in [Dockerfile](https://github.com/larpp/Nanoscale_Single-vesicle_Analysis/blob/main/ConvNeXt/Dockerfile).

To use Focal loss, focal_loss_torch must be installed.
```
pip install focal_loss_torch
```

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

If you use a single GPU, use the following command.

```
python main.py \
--model convnext_small --drop_path 0.1 \
--batch_size 32 --lr 4e-3 --update_freq 4 \
--model_ema true --model_ema_eval true \
--data_path dest \
--output_dir out \
--resume convnext_small_1k_224_ema.pth \
--nb_classes 2
```

## Evaluation

```
python -m torch.distributed.launch --nproc_per_node=2 main.py \
--model convnext_small --eval true \
--resume  out/checkpoint-best.pth \
--input_size 224 --drop_path 0.2 \
--batch_size 32 \
--data_path dest \
--nb_classes 2
```

If you use a single GPU, use the following command.
```
python main.py \
--model convnext_small --eval true \
--resume  out/checkpoint-best.pth \
--input_size 224 --drop_path 0.2 \
--batch_size 32 \
--data_path dest \
--nb_classes 2
```

## Inference

```
python main.py \
--model convnext_small \
--inference True \
--resume  out/checkpoint-best.pth \
--input_size 224 --drop_path 0.2 \
--batch_size 1 \
--data_path <inference data path> \
--csv_path <csv path> \
--nb_classes 2 \
--results_dir <results path>
```

# Acknowledgement
https://github.com/facebookresearch/ConvNeXt?tab=readme-ov-file
