# MiniMax H3 Director (Qwen Prompt)

基于 [AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) 导演台的 fork，整合本地 Qwen3.8-27B-VL GGUF 提示词优化器。

> **上游来源**：本仓库的 Qwen 提示词优化功能源自独立节点 [MiniMaxH3-Prompt-Optimizer](https://github.com/PigHoush/MiniMaxH3-Prompt-Optimizer)

## 这是什么

在 MiniMax H3 Director 导演台的提示词框旁边加了一排 Qwen 优化按钮，不用切到外部工具，直接在导演台里一键把大白话润色成 H3 规范提示词。

### 功能

- **一键优化**：全局提示词、每段提示词、batch 面板都有 ✨优化 按钮
- **视觉理解**：Qwen 是多模态模型，导演台挂的参考图自动喂进去，优化时真的看得到参考画面
- **时长感知**：自动把当前片段的秒数传给 Qwen，生成的提示词节奏匹配目标时长
- **实时预览**：所有视频模式都支持采样实时画面
- **模型自选**：把 .gguf 丢进 `engine/models/` 就能在下拉里选
- **不抢显存**：优化器用子进程跑，跑完自动退出，不跟生成模型抢 GPU

## 安装

### 前置
- 已安装 [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- 已安装 [MiniMax H3 Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director) 导演台节点
- 需要 llama.cpp 支持的 Qwen GGUF 模型

### 步骤

1. clone 这个仓库
2. 把 `ComfyUI/custom_nodes/` 里的两个文件夹复制到你的 ComfyUI custom_nodes 目录
3. 把 `engine/` 目录放到你方便的位置，改 `engine/config.json` 里的模型路径
4. 安装 Python 依赖：`pip install llama-cpp-python`
5. 启动 ComfyUI，在导演台提示词框上方就能看到 Qwen 控件栏

## 模型

需要自己下载，放在 `engine/models/` 下：
- Qwen3.8-27B-UD-Q2_K_XL.gguf（快速版 ~9.9G）
- Qwen3.8-27B-UD-Q3_K_XL.gguf（高质量版 ~12.5G）
- mmproj-Qwen3.8-27B-F16.gguf（视觉投影 ~0.9G）

## 更新日志

详见 [CHANGELOG.md](CHANGELOG.md)

## 致谢

- 原导演台：[AIMixer/ComfyUI_MiniMaxH3_Director](https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director)
- 提示词引擎：Qwen3.8 / llama.cpp
