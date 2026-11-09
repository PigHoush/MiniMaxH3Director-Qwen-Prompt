# -*- coding: utf-8 -*-
"""ComfyUI 独立节点：MiniMax H3 提示词优化器

薄封装：真正的推理由引擎脚本用 D:\\H3ComfyUI\\python 起子进程完成，
所以这里不 import llama_cpp / torch，不会和 ComfyUI 的 OpenMP、显存调度打架。
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 本文件位于 <root>\ComfyUI\custom_nodes\ComfyUI_H3PromptOptimizer\nodes.py
# parents[0]=ComfyUI_H3PromptOptimizer, [1]=custom_nodes, [2]=ComfyUI, [3]=<root>
_ROOT = Path(__file__).resolve().parents[3]
ENGINE_DIR = _ROOT / "engine"
ENGINE_PY = _ROOT / "python" / "python.exe"
ENGINE_SCRIPT = ENGINE_DIR / "h3_prompt_optimizer.py"
ENGINE_CONFIG = ENGINE_DIR / "config.json"

TASKS = ["T2VA", "I2VA", "L2VA", "FL2VA", "REF2VA", "HYBRID"]
CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _scan_models() -> list[tuple[str, str, str]]:
    """返回 [(下拉显示名, 主模型 id, mmproj id)], 只在导入时扫一次。"""
    import os as _os

    try:
        cfg = json.loads(ENGINE_CONFIG.read_text(encoding="utf-8"))
    except Exception:
        return []
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for root in cfg.get("model_dirs") or []:
        rp = Path(root)
        if not rp.is_dir():
            continue
        files: list[Path] = []
        # 用 os.walk 并 followlinks=True，才能跟进 models_e 这种目录联接
        for base, _dirs, names in _os.walk(str(rp), followlinks=True):
            for n in names:
                if n.lower().endswith(".gguf"):
                    files.append(Path(base) / n)
        files.sort(key=lambda p: str(p).lower())
        for f in files:
            if "mmproj" in f.name.lower():
                continue
            try:
                rel = f.relative_to(rp).as_posix()
            except ValueError:
                continue
            if rel in seen:
                continue
            seen.add(rel)
            mm = ""
            sibs = sorted(
                (p for p in f.parent.iterdir()
                 if p.is_file() and "mmproj" in p.name.lower() and p.suffix.lower() == ".gguf"),
                key=lambda p: str(p).lower(),
            )
            if sibs:
                try:
                    mm = sibs[0].relative_to(rp).as_posix()
                except ValueError:
                    mm = ""
            label = f.stem + (f"  ·  {Path(mm).name}" if mm else "  ·  （无 mmproj）")
            out.append((label, rel, mm))
    return out


MODELS = _scan_models()


def _save_images(images, work_dir: Path) -> list[Path]:
    """把 IMAGE 张量落成 png，顺序即 <picture 1..N>。"""
    if images is None:
        return []
    import numpy as np
    from PIL import Image

    paths = []
    arr = images
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        batch = [arr[i] for i in range(arr.shape[0])]
    else:
        batch = [arr]
    for i, frame in enumerate(batch, start=1):
        frame = np.clip(frame, 0.0, 1.0)
        if frame.shape[-1] > 3:
            frame = frame[..., :3]
        img = Image.fromarray((frame * 255.0).round().astype(np.uint8))
        p = work_dir / f"ref_{i}.png"
        img.save(p)
        paths.append(p)
    return paths


def _to_jpeg(frame, path) -> None:
    import numpy as np
    from PIL import Image

    arr = frame
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr, 0.0, 1.0)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    Image.fromarray((arr * 255.0).round().astype(np.uint8)).save(path, format="JPEG", quality=90)


def _save_video_frames(video, work_dir: Path, prefix: str = "video1") -> tuple[list[Path], int, float]:
    """VIDEO 类型 → 抽 3 帧（首 / 中 / 尾）。返回 (帧路径, 总帧数, fps)。"""
    if video is None:
        return [], 0, 0.0
    frames = None
    fps = 0.0
    try:
        if hasattr(video, "get_components"):
            comps = video.get_components()
            frames = comps.images
            try:
                fps = float(comps.frame_rate)
            except (TypeError, ValueError):
                fps = 0.0
        elif hasattr(video, "images"):
            frames = video.images
    except Exception as exc:  # noqa: BLE001
        print(f"[H3提示词优化] 读取视频失败: {exc}")
        return [], 0, 0.0
    if frames is None:
        return [], 0, 0.0

    import numpy as np

    arr = frames
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[None, ...]
    total = int(arr.shape[0])
    if total == 0:
        return [], 0, fps
    picks = sorted({0, total // 2, total - 1})
    paths = []
    for i, idx in enumerate(picks, start=1):
        p = work_dir / f"{prefix}_{i}.jpg"
        _to_jpeg(arr[idx], p)
        paths.append(p)
    return paths, total, fps


def _audio_note(audio) -> str:
    """AUDIO 类型只登记时长/采样率，音频内容原包不传输。"""
    if not isinstance(audio, dict):
        return "（内容未传输，仅作占位参考）"
    wav = audio.get("waveform")
    rate = audio.get("sample_rate")
    try:
        import numpy as np

        shape = np.asarray(wav.cpu() if hasattr(wav, "cpu") else wav).shape
        dur = float(shape[-1]) / float(rate)
        return f"时长约 {dur:.2f} 秒，采样率 {int(rate)} Hz（内容未传输，仅作占位参考）"
    except Exception:  # noqa: BLE001
        return "（内容未传输，仅作占位参考）"


class H3PromptOptimizer:

    CATEGORY = "MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("优化后的提示词", "保存路径")
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        labels = [m[0] for m in MODELS] or ["（未找到模型，检查 engine\\config.json 的 model_dirs）"]
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "dynamicPrompts": False}),
                "task": (TASKS, {"default": "T2VA"}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 60.0, "step": 1.0}),
                "output_language": (["中文", "English"], {"default": "中文"}),
                "max_tokens": ("INT", {"default": 4096, "min": 512, "max": 8192, "step": 256}),
                "model": (labels, {"default": labels[0]}),
                "copy_to_clipboard": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                **{f"image_{i}": ("IMAGE",) for i in range(1, 10)},
                **{f"video_{i}": ("VIDEO",) for i in range(1, 4)},
                **{f"audio_{i}": ("AUDIO",) for i in range(1, 4)},
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # 同样的输入可以命中缓存；想强制重跑就点一次改变 duration 或取消缓存
        return float("nan")

    def run(self, text, task, duration, output_language, max_tokens, model,
            copy_to_clipboard, **kwargs):
        if not str(text or "").strip():
            raise RuntimeError("提示词为空：先把你的想法写在 text 里（可多行）。")
        if not ENGINE_PY.is_file():
            raise RuntimeError(f"找不到独立引擎: {ENGINE_PY}")
        if not ENGINE_SCRIPT.is_file():
            raise RuntimeError(f"找不到主脚本: {ENGINE_SCRIPT}")

        task = str(task or "T2VA").strip().upper()
        if task not in TASKS:
            raise RuntimeError(f"任务类型只能是 {', '.join(TASKS)}，收到: {task}")

        # 模型只用节点自带下拉，不再接受外部连线覆盖
        target = str(model or "")
        rel, mmproj = "", ""
        for label, r, m in MODELS:
            if target == label or target == r or Path(r).name == target:
                rel, mmproj = r, m
                break

        work = Path(tempfile.mkdtemp(prefix="h3po_"))
        prompt_file = work / "prompt.txt"
        prompt_file.write_text(str(text), encoding="utf-8")
        out_file = work / "optimized.txt"

        cmd = [str(ENGINE_PY), str(ENGINE_SCRIPT),
               "--prompt-file", str(prompt_file),
               "--out", str(out_file),
               "--task", str(task),
               "--duration", f"{float(duration):g}",
               "--lang", str(output_language),
               "--max-tokens", str(int(max_tokens))]
        if rel:
            cmd += ["--model", rel]
        if mmproj:
            cmd += ["--mmproj", mmproj]
        # 9 个图片插槽：按顺序收集，每个插槽自身也可能是 batch，拍平后统一编号 <picture N>
        img_paths: list[Path] = []
        for i in range(1, 10):
            slot = kwargs.get(f"image_{i}")
            if slot is None:
                continue
            for p in _save_images(slot, work):
                img_paths.append(p)
        spec = []
        for idx, p in enumerate(img_paths, start=1):
            spec.append({"label": f"<picture {idx}>", "kind": "image", "images": [str(p)]})

        video_count = 0
        for i in range(1, 4):
            slot = kwargs.get(f"video_{i}")
            if slot is None:
                continue
            frame_paths, total_frames, fps = _save_video_frames(slot, work, prefix=f"video{i}")
            if not frame_paths:
                continue
            video_count += 1
            note = f"共 {total_frames} 帧" + (f"，{fps:.2f} fps" if fps else "") + f"，采样 {len(frame_paths)} 帧"
            spec.append({"label": f"<video {i}>", "kind": "video",
                         "images": [str(p) for p in frame_paths], "note": note})

        audio_count = 0
        for i in range(1, 4):
            slot = kwargs.get(f"audio_{i}")
            if slot is None:
                continue
            audio_count += 1
            spec.append({"label": f"<audio {i}>", "kind": "audio", "note": _audio_note(slot)})

        if spec:
            media_file = work / "media.json"
            media_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            cmd += ["--media-file", str(media_file)]
        if copy_to_clipboard:
            cmd += ["--copy"]

        print(f"[H3提示词优化] 任务={task} 时长={duration:g}s 模型={rel or '配置默认'} "
              f"图={len(img_paths)} 视频={video_count} 音频={audio_count}")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                              creationflags=CREATE_NO_WINDOW, encoding="utf-8",
                              errors="replace")
        if proc.returncode != 0 or not out_file.is_file():
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
            raise RuntimeError("提示词优化失败：\n" + "\n".join(tail))
        result = out_file.read_text(encoding="utf-8").strip()
        print(f"[H3提示词优化] 完成，{len(result)} 字符，已保存 {out_file}")
        return {"ui": {"text": [result]}, "result": (result, str(out_file))}


class H3ModelPicker:
    """独立选模型节点：只负责挑模型，把模型 id 和 mmproj id 传给优化器。"""

    CATEGORY = "MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("模型ID", "mmproj")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        labels = [m[0] for m in MODELS] or ["（未找到模型，检查 engine\\config.json 的 model_dirs）"]
        default = next(
            (m[0] for m in MODELS if m[1].endswith("Qwen3.8-27B-UD-Q2_K_XL.gguf")),
            labels[0],
        )
        return {"required": {"model": (labels, {"default": default})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, model):
        for label, r, m in MODELS:
            if label == model:
                print(f"[H3模型选择] {r}  |  mmproj={m or '无'}")
                return (r, m or "")
        raise RuntimeError(f"未知模型: {model}")


class H3TaskPicker:
    """独立任务类型节点：挑 T2VA / I2VA / L2VA / FL2VA 等，并给出参考图要求提示。"""

    CATEGORY = "MiniMax H3"
    FUNCTION = "run"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("任务类型", "参考输入说明")
    OUTPUT_NODE = False

    HINTS = {
        "T2VA": "纯文本生视频，不需要参考图",
        "I2VA": "首帧：接 1 张图，作为 0.00s 开场锚点",
        "L2VA": "尾帧：接 1 张图，作为结尾锚点，倒推开场",
        "FL2VA": "首尾帧：接 2 张图，picture 1 是首帧、picture 2 是尾帧",
        "REF2VA": "多参考：接参考图/视频，输出六段保留度分析",
        "HYBRID": "混合参考：图片+视频+音频可同接",
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"task": (TASKS, {"default": "T2VA"})}}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def run(self, task):
        task = str(task or "T2VA").upper()
        if task not in TASKS:
            raise RuntimeError(f"任务类型只能是 {', '.join(TASKS)}")
        print(f"[H3任务类型] {task} — {self.HINTS.get(task, '')}")
        return (task, self.HINTS.get(task, ""))


NODE_CLASS_MAPPINGS["H3PromptOptimizer"] = H3PromptOptimizer
NODE_CLASS_MAPPINGS["H3ModelPicker"] = H3ModelPicker
NODE_CLASS_MAPPINGS["H3TaskPicker"] = H3TaskPicker
NODE_DISPLAY_NAME_MAPPINGS["H3PromptOptimizer"] = "MiniMax H3 提示词优化器"
NODE_DISPLAY_NAME_MAPPINGS["H3ModelPicker"] = "MiniMax H3 模型选择"
NODE_DISPLAY_NAME_MAPPINGS["H3TaskPicker"] = "MiniMax H3 任务类型"


# ---------------------------------------------------------------------------
# HTTP 路由：给导演台 Pro 版的"提示词增强"按钮调用
# POST /h3_optimize
# body: {text, task?, output_language?, images_b64?, duration?, max_tokens?}
# 返回: {text: "..."} 或 {error: "..."}
# ---------------------------------------------------------------------------
async def _h3_optimize_handler(request):
    import base64
    from aiohttp import web

    try:
        data = await request.json()
    except Exception as exc:
        return web.json_response({"error": f"invalid JSON: {exc}"}, status=400)

    text = (data.get("text") or "").strip()
    if not text:
        return web.json_response({"error": "empty text"}, status=400)

    task = str(data.get("task") or "T2VA").strip().upper()
    if task not in TASKS:
        # 导演台传 t2v/i2v/fl2v/r2v/v2v/rv2v，做一次映射
        mapping = {
            "T2V": "T2VA", "I2V": "I2VA", "FL2V": "FL2VA",
            "R2V": "REF2VA", "RV2V": "REF2VA", "L2V": "L2VA",
            "HYBRID": "HYBRID",
        }
        task = mapping.get(task, "T2VA")

    output_language = str(data.get("output_language") or "中文")
    duration = float(data.get("duration") or 10.0)
    max_tokens = int(data.get("max_tokens") or 4096)

    work = Path(tempfile.mkdtemp(prefix="h3http_"))
    prompt_file = work / "prompt.txt"
    prompt_file.write_text(text, encoding="utf-8")
    out_file = work / "optimized.txt"

    cmd = [str(ENGINE_PY), str(ENGINE_SCRIPT),
           "--prompt-file", str(prompt_file),
           "--out", str(out_file),
           "--task", task,
           "--duration", f"{duration:g}",
           "--lang", output_language,
           "--max-tokens", str(max_tokens)]

    # 模型快捷选择：q2 / q3 对应两个已知 gguf；也可传完整相对路径
    model_alias = str(data.get("model") or "").strip()
    if model_alias:
        ALIAS = {
            "q2": "Q38-27B-G-UD/Qwen3.8-27B-UD-Q2_K_XL.gguf",
            "q3": "Q38-27B-G-UD/Qwen3.8-27B-UD-Q3_K_XL.gguf",
            "q2k": "Q38-27B-G-UD/Qwen3.8-27B-UD-Q2_K_XL.gguf",
            "q3k": "Q38-27B-G-UD/Qwen3.8-27B-UD-Q3_K_XL.gguf",
        }
        resolved = ALIAS.get(model_alias.lower(), model_alias)
        cmd += ["--model", resolved]

    # 导演台传过来的参考图（base64，无 data: 前缀或带都接受）
    imgs = data.get("images_b64") or []
    spec = []
    for idx, b64 in enumerate(imgs, start=1):
        if not b64:
            continue
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            raw = base64.b64decode(b64)
        except Exception:
            continue
        p = work / f"ref_{idx}.jpg"
        p.write_bytes(raw)
        spec.append({"label": f"<picture {idx}>", "kind": "image", "images": [str(p)]})
    if spec:
        media_file = work / "media.json"
        media_file.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
        cmd += ["--media-file", str(media_file)]

    # 参考音频/视频文件名作为上下文拼进提示词（Qwen 读不了文件本身，但知道挂了什么素材有助于写提示词）
    audio_refs = data.get("audio_refs") or []
    video_refs = data.get("video_refs") or []
    extra_ctx = ""
    if audio_refs:
        extra_ctx += "\n[参考音频素材]\n" + "\n".join(f"- {a}" for a in audio_refs)
    if video_refs:
        extra_ctx += "\n[参考视频素材]\n" + "\n".join(f"- {v}" for v in video_refs)
    if extra_ctx:
        text_with_ctx = text + "\n\n" + extra_ctx + "\n（写提示词时请结合以上参考素材描述对应的音频/视频元素，但不要编造素材内容）"
        prompt_file.write_text(text_with_ctx, encoding="utf-8")

    print(f"[H3提示词优化/HTTP] task={task} lang={output_language} images={len(spec)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900,
                          creationflags=CREATE_NO_WINDOW, encoding="utf-8",
                          errors="replace")
    if proc.returncode != 0 or not out_file.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
        return web.json_response(
            {"error": "engine failed", "detail": "\n".join(tail)}, status=500
        )
    result = out_file.read_text(encoding="utf-8").strip()
    print(f"[H3提示词优化/HTTP] OK, {len(result)} chars")
    return web.json_response({"text": result})


async def _h3_models_handler(request):
    """GET /h3_models — 扫描 engine/models 下所有 .gguf（排除 mmproj），返回列表"""
    from aiohttp import web
    models_root = Path(__file__).resolve().parents[2] / "engine" / "models"
    out = []
    if models_root.is_dir():
        for p in sorted(models_root.rglob("*.gguf")):
            name = p.name.lower()
            if "mmproj" in name:
                continue
            rel = p.relative_to(models_root).as_posix()
            size_gb = round(p.stat().st_size / 1e9, 1)
            out.append({"id": rel, "name": p.stem, "size_gb": size_gb})
    return web.json_response({"models": out})


try:
    from aiohttp import web as _web  # noqa: F401
    from server import PromptServer  # type: ignore

    PromptServer.instance.app.add_routes([
        _web.post("/h3_optimize", _h3_optimize_handler),
        _web.get("/h3_models", _h3_models_handler),
    ])
except Exception as _route_exc:  # pragma: no cover
    import logging as _lg
    _lg.getLogger("ComfyUI-H3PromptOptimizer").warning(
        "/h3_optimize route not registered: %s", _route_exc
    )
