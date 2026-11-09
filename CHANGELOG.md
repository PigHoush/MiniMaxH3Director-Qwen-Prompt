# MiniMax H3 Director (Qwen Prompt) — 更新日志

> 基于 MiniMax H3 Director 导演台 fork，整合本地 Qwen3.8-27B-VL GGUF 提示词优化器。

---

## v1.1.0 — 2026-11-09

### 修复

**时长控制**
之前用导演台做短剧，每段提示词要自己判断写多少字、动作描述多长，经常出现提示词写多了结果视频时长不够、或者写少了画面空。Qwen 优化器虽然能把提示词润色成 H3 规范格式，但生成出来的内容跟导演台设定的秒数完全没关系——前端三处调用（全局/分段/batch）都漏传了 duration 参数，Qwen 不知道这段要几秒，自然写出来的节奏不对。现在三处都自动取当前片段的秒数传给引擎，Qwen 在 system prompt 里拿到 `duration=X.Xs`，生成的提示词会严格控制在这个时长内能演完的信息量。

**实时预览**
导演台本身有实时采样预览功能，但之前 `needsLiveSamplePanel()` 方法里写死了只对 fl2v、v2v、mv2v 等少数模式显示预览面板。平时做短剧主要用 t2v 和 i2v，结果点了"实时预览"按钮后后端一直在推帧数据（websocket 事件都发出来了），前端面板却不渲染，白等半天看不到画面。改成只要开了开关就显示面板，所有视频模式都能看到采样过程。

### 改动文件
| 文件 | 改动 |
|------|------|
| `web/js/minimax_timeline.js` | 两处 fetch 加 duration（分段取 `selectedSegIndex.durationSec`，全局取 `timeline.durationSec`）；`needsLiveSamplePanel()` 改为开了开关就返回 true（image batch 除外） |
| `web/js/minimax_image_batch.js` | batch fetch 加 duration（取 `seg.durationSec`） |
| `__init__.py` | 节点显示名统一为 `MiniMaxH3Director-Qwen-Prompt` |
| `启动.bat` | 加 `--auto-launch`，启动完自动开浏览器 |
| 文件夹 | `ComfyUI_MiniMaxH3_Director_Pro` → `ComfyUI_MiniMaxH3Director_QwenPrompt`，节点底部标签同步 |

---

## v1.0.0 — 2026-11-09

### 目录结构
```
D:\MiniMaxH3Director-Qwen-Prompt\
├── ComfyUI\
│   ├── main.py
│   ├── models\                    ← junction → D:\ComfyUI-MiniMaxH3Director\ComfyUI\models
│   └── custom_nodes\
│       ├── ComfyUI_MiniMaxH3Director_QwenPrompt\   ← 导演台 fork
│       │   ├── __init__.py
│       │   └── web\js\
│       │       ├── minimax_timeline.js
│       │       └── minimax_image_batch.js
│       └── ComfyUI_H3PromptOptimizer\       ← Qwen 优化节点
│           └── nodes.py
├── engine\
│   ├── h3_prompt_optimizer.py     ← Qwen 引擎（子进程）
│   ├── config.json
│   └── models\Q38-27B-G-UD\
│       ├── Qwen3.8-27B-UD-Q2_K_XL.gguf   (9.9G, 快)
│       ├── Qwen3.8-27B-UD-Q3_K_XL.gguf   (12.5G, 准)
│       └── mmproj-Qwen3.8-27B-F16.gguf   (0.9G, 视觉投影)
├── python\
└── 启动MiniMaxH3Director-Qwen-Prompto.bat
```

### 为什么做这个 fork

原来写提示词要在导演台和外部 LLM 之间来回复制粘贴，参考图还要手动描述，做一段分镜切好几个窗口。导演台自带的 LLM 增强又依赖外部 API 不好用。所以 fork 出来直接把本地 Qwen3.8-27B-VL 整合进去：本地跑不花钱、子进程跑完释放显存不抢 GPU、参考图直接喂给视觉模型真的看得懂、任务模式自动跟随导演台。

### 新增功能

**Qwen 优化控件栏**
加在三处提示词框正上方——全局提示词、每个分段的提示词、batch 面板。横排一排控件，不占额外高度：
- 模型下拉：自动扫描 `engine\models\` 下所有 .gguf 文件，以后加新模型直接丢文件夹里就行，不用改代码
- 任务下拉：自动跟随导演台当前模式填好，也可以手动改成别的
- ✨优化：一键调用，把当前提示词润色成 H3 规范格式
- ↩复原：不满意一键退回原文
- ✕取消：生成到一半不想等了直接中断

**参考素材感知**
- 导演台挂的参考图自动截出来转 base64 喂给 Qwen 的视觉模块，优化时会参考图里的人物/场景/构图
- 参考视频和音频虽然不能直接"看"，但文件名会作为上下文传给 Qwen，让它知道这段参考的是什么
- 空提示词也能优化——根据任务模式自动生成一段，不用先写个开头

### 改动文件
| 文件 | 改动 |
|------|------|
| `__init__.py` | 注册 `/h3_optimize` 和 `/h3_models` 路由；节点显示名 |
| `web/js/minimax_timeline.js` | 全局+每段提示词框上方插入 Qwen 控件栏；收集参考图转 base64 |
| `web/js/minimax_image_batch.js` | batch 面板同样插入控件栏；参考图传递 |
| `custom_nodes/ComfyUI_H3PromptOptimizer/nodes.py` | `/h3_optimize` 接收 prompt/task/model/duration/images，调引擎子进程；`/h3_models` 扫描模型目录 |
| `engine/h3_prompt_optimizer.py` | Qwen 推理引擎，子进程启动 llama.cpp 加载 GGUF，生成完退出 |
