# A text-supervised open-vocabulary semantic segmentation method with spatial-semantic feature fusion for remote sensing imagery
This repository is the official implementation of "A text-supervised open-vocabulary semantic segmentation method with spatial-semantic feature fusion for remote sensing imagery".

## Prepare dataset
We provide the [GID dataset](https://drive.google.com/file/d/1g30ldLFhJaqPTTxPHIZYRWnREwFKZf77/view?usp=sharing) divided into 256✕256 as example data to show the preprocessing process.

1. Using [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) or other MLLMS to generate textual descriptions for remote sensing images. We provide pre-generated textual descriptions of 15 words and 50 words in length, as given in [train_caption_15words.csv](https://drive.google.com/file/d/1g30ldLFhJaqPTTxPHIZYRWnREwFKZf77/view?usp=sharing) and [train_caption_50words.csv](https://drive.google.com/file/d/1g30ldLFhJaqPTTxPHIZYRWnREwFKZf77/view?usp=sharing).
2. Using large language models such as Qwen3-VL, DeepSeek, Gemini, and GPT to extract remote sensing object entities from the textual descriptions, as shown in [geo_entities.py](data_tools/geo_entities.py).
3. Using [data_process.py](data_tools/data_process.py) to filter GID dataset using these entities. This will generate 8 sub-files in the `subsets/` directory.
```shell
python -m data_tools.data_process --mode filter --srcdir /path/to/your/gid/train_caption_15words.csv --processors 8
```
4. Next, merge these sub-files into a single metafile (and optionally delete the sub-files by passing `--remove_subfiles True`).
```shell
python -m data_tools.data_process --mode merge --dstdir /path/to/your/gid/subsets/
```
5. Construct cross-image pairs based on the filtered data. The generated metafile is automatically saved to `/path/to/your/gid/subsets/gid_filtered_subset_pair.csv`. This metafile can be used for training the model. We give the pre-generated `gid_filtered_subset_pair.csv` in [GID dataset](https://drive.google.com/file/d/1g30ldLFhJaqPTTxPHIZYRWnREwFKZf77/view?usp=sharing).
```shell
python -m data_tools.data_process --mode makepair --metafile /path/to/your/gid/subsets/gid_filtered_subset.csv
```
6. Modify `data.img_dir` and `data.metas_path` in [config_train.yaml](configs/config_train.yaml). The training CSV should contain `image_id`, `caption`, `pairindex`, and `pairentity` columns.

Note: The current `data_process.py` calls `.keys()` on `Entities_GID_15`, but [geo_entities.py](data_tools/geo_entities.py) defines it as a list. This mismatch must be fixed before running the preprocessing commands above.

## Prepare model
1. Download DINO pretrained weights and specify the model path.
```shell
https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth
```
2. Set `model.dino_path` in [config_train.yaml](configs/config_train.yaml) to the downloaded weights. Add the same setting under `model` in [config_fusion.yaml](configs/config_fusion.yaml) before inference; the fusion code requires it, but the key is not currently in that file.
```shell
dino_path: '/path/to/your/dino_vitbase16_pretrain.pth'
```

## Group-level image-text contrastive learning
Run [main_train.py](main_train.py) from the repository root to perform text-supervised contrastive training. It uses all visible CUDA devices and saves the trained model to `checkpoint/net.pt`. We have provided a pre-trained coarse-grained model, [net_GID.pth](https://drive.google.com/file/d/1L9gXQ2sZ_A0z8240g5VIEJjJ8fZJ1v1A/view?usp=drive_link), which was trained on the GID dataset.

## Spatial-semantic feature fusion
Run [main_fusion.py](main_fusion.py) from the repository root to perform fine-level open-vocabulary semantic segmentation. Set `model.group_vit.checkpoint` in [config_fusion.yaml](configs/config_fusion.yaml) to the trained model. The example reads `imgs/32_img.png`, uses the text vocabulary in [gid.json](datasets/gid.json), and saves its result to `visualization/out.png`. To use another image, change the path in `main_fusion.py`; to use another vocabulary, edit `gid.json`. For evaluation, set `data.val.img_dir` and `data.val.label_dir` in `config_fusion.yaml` and run [main_fusion_evaluate.py](main_fusion_evaluate.py).

## Acknowledgement
We would like to acknowledge the contributions of public projects, such as [GroupViT](https://github.com/NVlabs/GroupViT), [ClearCLIP](https://github.com/mc-lan/ClearCLIP), [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP), and [OVSegmentor](https://github.com/Jazzcharles/OVSegmentor) , whose code has been utilized in this repository.
