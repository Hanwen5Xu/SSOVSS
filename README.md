# A text-supervised open-vocabulary semantic segmentation method with spatial-semantic feature fusion for remote sensing images
This repository is the official implementation of "A text-supervised open-vocabulary semantic segmentation method with spatial-semantic feature fusion for remote sensing images".

## Prepare dataset
The preparation process of the dataset refers to [OVSegmentor](https://github.com/Jazzcharles/OVSegmentor). We provide the [GID dataset](https://drive.google.com/file/d/1RbSsL9_7HPPHkt_px1pYoBM9XfYwJUHV/view?usp=drive_link) divided into 256✕256 as example data to show the preprocessing process.

1. Using [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) or other MLLMS to generate textual descriptions for remote sensing images. We provide pre-generated textual descriptions of 15 words and 50 words in length, as given in [train_caption_15words.csv](https://drive.google.com/file/d/1RbSsL9_7HPPHkt_px1pYoBM9XfYwJUHV/view?usp=drive_link) and [train_caption_50words.csv](https://drive.google.com/file/d/1RbSsL9_7HPPHkt_px1pYoBM9XfYwJUHV/view?usp=drive_link).
2. Using large language models such as Qwen3-VL, DeepSeek, Gemini, and GPT to extract remote sensing object entities from the textual descriptions, as shown in [geo_entities.py](data_tools/geo_entities.py).
3. Using [data_process.py](data_tools/data_process.py) to filter GID dataset using these entities. This will generate 8 sub-files in subset/ directory. 
```shell 
cd data_tools
python data_process.py --mode filter --srcdir /path/to/your/train_caption_20words.csv --processor 8
```
4. Next, merge these sub-files into a single metafile (and optionally delete the sub-files by passing --remove_subfiles=True).
```shell
python data_process.py --mode merge --dstdir /path/to/your/gid/subsets/ --remove_subfiles True
```
5. Construct cross-image pairs based on the filtered data. The generated metafile is automatically saved to /path/to/your/gid_filtered_subset_pair.csv. This metafile can be used for training the model. We give the pre-generated gid_filtered_subset_pair.csv in [GID dataset](https://drive.google.com/file/d/1RbSsL9_7HPPHkt_px1pYoBM9XfYwJUHV/view?usp=drive_link).
```shell
python data_process.py --mode makepair --metafile /path/to/your/gid_filtered_subset.csv
```
6. Modify the img_dir and metas_path in [config.yaml](configs/config.yaml).

## Prepare model
1. Download DINO pretrained weights and specify the model path.
```shell
https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```
2. Change the configs [config.yaml](configs/config.yaml) by specifying the checkpoint path.
```shell
dino_path: '/path/to/your/dino_vitbase16_pretrain.pth'
```

## Group-level image-text contrastive learning
Using [main_train.py](main_train.py) to perform text-supervised contrastive training, and the trained coarse-grained model is saved to checkpoint. We have provided a pre-trained coarse-grained model, [net_GID.pth](https://drive.google.com/file/d/1L9gXQ2sZ_A0z8240g5VIEJjJ8fZJ1v1A/view?usp=drive_link), which was trained on the GID dataset.

## Fpatial-semantic feature fusion
Using [main_fusion.py](main_fusion.py) to perform fine-level open-vocabulary semantic segmentation. Users can input custom text vocabularies to execute the inference.

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [GroupViT](https://github.com/NVlabs/GroupViT), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP), and [OVSegmentor](https://github.com/Jazzcharles/OVSegmentor) , whose code has been utilized in this repository.
