# Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation
### [Project Page](https://promptda.github.io/) | [Paper](https://promptda.github.io/assets/main_paper_with_supp.pdf) | [Hugging Face Demo](https://huggingface.co/spaces/depth-anything/PromptDA) | [Interactive Results](https://promptda.github.io/interactive.html) | [Data](https://promptda.github.io/)

> Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation  
> [Haotong Lin](https://haotongl.github.io/),
[Sida Peng](https://pengsida.net/),
[Jingxiao Chen](https://scholar.google.com/citations?user=-zs1V28AAAAJ),
[Songyou Peng](https://pengsongyou.github.io/),
[Jiaming Sun](https://jiamingsun.me/),
[Minghuan Liu](https://minghuanliu.com/),
[Hujun Bao](http://www.cad.zju.edu.cn/home/bao/),
[Jiashi Feng](https://scholar.google.com/citations?user=Q8iay0gAAAAJ),
[Xiaowei Zhou](https://www.xzhou.me/),
[Bingyi Kang](https://bingykang.github.io/)  
> CVPR 2025

![teaser](assets/teaser.gif)

## 📰 News
- [PromptDA++](assets/paper.pdf) paper is available. Code and models will be released, please stay tuned.
- Releasing [ScanNet++ ZipNeRF Reconstruction Depth Results](https://huggingface.co/datasets/haotongl/scannetpp_zipnerf/tree/main)


## 🛠️ Installation

<details> <summary> Setting up the environment </summary>

```bash
git clone https://github.com/DepthAnything/PromptDA.git
cd PromptDA
pip install -r requirements.txt
pip install -e .
sudo apt install ffmpeg  # for video generation
```
</details>
<details> <summary> Pre-trained Models </summary>

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Prompt-Depth-Anything-Large | 340M | [Download](https://huggingface.co/depth-anything/prompt-depth-anything-vitl/resolve/main/model.ckpt) |
| Prompt-Depth-Anything-Small | 25.1M | [Download](https://huggingface.co/depth-anything/prompt-depth-anything-vits/resolve/main/model.ckpt) |
| Prompt-Depth-Anything-Small-Transparent | 25.1M | [Download](https://huggingface.co/depth-anything/prompt-depth-anything-vits-transparent/resolve/main/model.ckpt) |

Only Prompt-Depth-Anything-Large is used to benchmark in our paper. Prompt-Depth-Anything-Small-Transparent is further fine-tuned 10K steps with [hammer dataset](https://github.com/Junggy/HAMMER-dataset) with our iPhone lidar simulation method to improve the performance on transparent objects.

</details>


## 🚀 Usage
<details> <summary> Example usage </summary>

```python
from promptda.promptda import PromptDA
from promptda.utils.io_wrapper import load_image, load_depth, save_depth

DEVICE = 'cuda'
image_path = "assets/example_images/image.jpg"
prompt_depth_path = "assets/example_images/arkit_depth.png"
image = load_image(image_path).to(DEVICE)
prompt_depth = load_depth(prompt_depth_path).to(DEVICE) # 192x256, ARKit LiDAR depth in meters

model = PromptDA.from_pretrained("depth-anything/prompt-depth-anything-vitl").to(DEVICE).eval()
depth = model.predict(image, prompt_depth) # HxW, depth in meters

save_depth(depth, prompt_depth=prompt_depth, image=image)
```
</details>


## 📸 Running on your own capture

You can use [Stray Scanner App](https://apps.apple.com/us/app/stray-scanner/id1557051662) to capture your own data, which requires iPhone 12 Pro or later Pro models, iPad 2020 Pro or later Pro models. We setup a [Hugging Face Space](https://huggingface.co/spaces/depth-anything/PromptDA) for you to quickly test our model. If you want to obtain video results, please follow the following steps.

<details> <summary> Testing steps </summary>

1. Capture a scene with the Stray Scanner App. (The charging port is preferred to face downward or to the right.)
2. Use the iPhone Files App to compress it into a zip file and transfer it to your computer. Here is an [example screen recording](https://haotongl.github.io/promptda/assets/ScreenRecording_12-16-2024.mp4).
3. Run the following commands to infer our model and generate the video results.
```bash
export PATH_TO_ZIP_FILE=data/8b98276b0a.zip # Replace with your own zip file path
export PATH_TO_SAVE_FOLDER=data/8b98276b0a_results # Replace with your own save folder path
python3 -m promptda.scripts.infer_stray_scan --input_path ${PATH_TO_ZIP_FILE} --output_path ${PATH_TO_SAVE_FOLDER}
python3 -m promptda.scripts.generate_video process_stray_scan --input_path ${PATH_TO_ZIP_FILE} --result_path ${PATH_TO_SAVE_FOLDER}
ffmpeg -framerate 60 -i ${PATH_TO_SAVE_FOLDER}/%06d_smooth.jpg  -c:v libx264 -pix_fmt yuv420p ${PATH_TO_SAVE_FOLDER}.mp4
```
</details>

## 🔍 專案深度分析（PromptDA）

### 1) 核心技術架構

PromptDA 的核心推理路徑可以概括為：

**RGB 影像 + Prompt 深度 → DINOv2 Encoder 特徵抽取 → DPT Decoder 多尺度融合 → Metric Depth 輸出**

- **Backbone（Encoder）**：`/promptda/promptda.py` 中以 `torch.hub.load(...)` 載入本地 DINOv2（`dinov2_{encoder}14`），並透過 `get_intermediate_layers(...)` 取出多層特徵。
- **Head（Decoder）**：`/promptda/model/dpt.py` 的 `DPTHead` 先做特徵投影與尺度對齊，再經過 4 層 refine/fusion block 輸出深度。
- **模型配置**：`/promptda/model/config.py` 定義 `vits/vitb/vitl/vitg` 的 `features/out_channels/layer_idxs`。
- **I/O 與推理流程**：`/promptda/utils/io_wrapper.py` + `/promptda/scripts/infer_stray_scan.py` 負責資料讀寫、批次推理與結果落盤。

### 2) Prompt 在深度預估中的角色

Prompt（通常是 ARKit LiDAR 的低解析度深度）在這個專案中是**條件引導訊號**，不是單純後處理：

- 在 `PromptDA.forward(...)`（`/promptda/promptda.py`）中，`prompt_depth` 是必填，先做 normalize，再交給 `DPTHead`。
- 在 `FeatureFusionDepthBlock`（`/promptda/model/blocks.py`）中，`prompt_depth` 會：
  1. 先插值到當前特徵尺度；
  2. 經 `resConfUnit_depth` 轉成特徵；
  3. 透過殘差加法與 RGB 路徑特徵融合。
- `resConfUnit_depth` 最後一層使用 `zero_module(...)` 零初始化，讓網路在訓練初期不會被 prompt 分支強行主導，融合強度可由訓練逐步學習。

### 3) 模型核心實作檔案

- **`/promptda/promptda.py`**  
  主模型類 `PromptDA`：模型組裝、checkpoint 載入、prompt normalize/denormalize、`forward/predict`。
- **`/promptda/model/dpt.py`**  
  `DPTHead`：多層 Transformer 特徵轉成空間特徵，並用 4 層 refinenet 做上採樣與融合。
- **`/promptda/model/blocks.py`**  
  `FeatureFusionDepthBlock` 是 PromptDA 的關鍵融合模組；同檔案也包含殘差與融合基礎積木。
- **`/promptda/model/config.py`**  
  各編碼器版本的結構超參數與抽層配置。

### 4) 相比原版 Depth Anything 的最大改進

相對於原版 Depth Anything（以單 RGB 輸入估計為主），PromptDA 最核心的改進是：

1. **加入 Prompt 深度條件化機制**：在 DPT 的多尺度融合階段直接注入 prompt 深度特徵（而非僅後處理）。
2. **面向高解析度 metric depth**：專案目標與流程明確聚焦於 4K resolution accurate metric depth estimation（見專案標題與推理流程）。
3. **工程化落地流程完整**：提供 Stray Scanner 資料推理、可視化與影片生成腳本，形成可直接實測的端到端管線。


## 👏 Acknowledgements
We thank the generous support from Prof. [Weinan Zhang](https://wnzhang.net/) for robot experiments, including the space, objects and the Unitree H1 robot. We also thank [Zhengbang Zhu](https://scholar.google.com/citations?user=ozatRA0AAAAJ), Jiahang Cao, Xinyao Li, Wentao Dong for their help in setting up the robot platform and collecting robot data.

## 📚 Citation
If you find this code useful for your research, please use the following BibTeX entry
```
@inproceedings{lin2024promptda,
  title={Prompting Depth Anything for 4K Resolution Accurate Metric Depth Estimation},
  author={Lin, Haotong and Peng, Sida and Chen, Jingxiao and Peng, Songyou and Sun, Jiaming and Liu, Minghuan and Bao, Hujun and Feng, Jiashi and Zhou, Xiaowei and Kang, Bingyi},
  journal={arXiv},
  year={2024}
}
```
