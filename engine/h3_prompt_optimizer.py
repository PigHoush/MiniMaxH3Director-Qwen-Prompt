# -*- coding: utf-8 -*-
"""独立引擎：MiniMax H3 提示词优化

本文件只做三件事：
1. 用空壳替掉 ComfyUI 的 folder_paths / server，再 import 原包的 prompt_optimizer.py；
2. 把原包的模型目录改到本环境的 models_e（ASCII 目录联接，避开 mtmd 中文路径坑），
   并用 os.walk(followlinks=True) 重新实现扫描（pathlib 的 rglob 不跟进目录联接）；
3. 跑一次完整的生成 + 后处理，结果写到 --out。

节点 ComfyUI_H3PromptOptimizer 就是以子进程方式调用本文件。
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import datetime as _dt
import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
PKG_DIR = VENDOR / "Goohai-MiniMax-H3_Integration"
TARGET = PKG_DIR / "prompt_optimizer.py"
CONFIG_PATH = ROOT / "config.json"

TASKS = ("T2VA", "I2VA", "L2VA", "FL2VA", "REF2VA", "HYBRID")
TASK_ALIASES = {
    "T2V": "T2VA", "I2V": "I2VA", "L2V": "L2VA", "FL2V": "FL2VA",
    "R2V": "REF2VA", "REF2V": "REF2VA",
}

DEFAULT_CONFIG = {
    "model_dirs": [str(ROOT.parent / "models_e"), str(ROOT.parent / "models")],
    "local_model": "",
    "local_mmproj": "",
    "task": "T2VA",
    "duration": 10,
    "output_language": "中文",
    "max_tokens": 4096,
    "image_max_edge": 1280,
    "out_dir": str(ROOT / "输出"),
}

# 让 llama.cpp 自带的 libomp 与 torch 的 libiomp 共存（子进程里保险起见也设上）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def _utf8_streams() -> None:
    for name in ("stdin", "stdout", "stderr"):
        stream = getattr(sys, name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# ---------------------------------------------------------------- 配置
def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.is_file():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                cfg.update({k: v for k, v in data.items() if v is not None})
        except (OSError, ValueError) as exc:
            print(f"[警告] 配置文件读取失败，使用默认值: {exc}")
    return cfg


# ---------------------------------------------------------------- 加载原包
def _install_shims() -> None:
    if "folder_paths" not in sys.modules:
        fp = types.ModuleType("folder_paths")
        fp.get_input_directory = lambda: str(ROOT)
        fp.get_output_directory = lambda: str(ROOT)
        fp.input_directory = str(ROOT)
        sys.modules["folder_paths"] = fp
    if "server" not in sys.modules:
        srv = types.ModuleType("server")

        class PromptServer:
            instance = None
            routes = None

        srv.PromptServer = PromptServer
        sys.modules["server"] = srv


def _walk_gguf(root: Path) -> list[Path]:
    found: list[Path] = []
    for base, _dirs, names in os.walk(str(root), followlinks=True):
        for name in names:
            if name.lower().endswith(".gguf"):
                found.append(Path(base) / name)
    return sorted(found, key=lambda p: str(p).lower())


def _bind_junction_safe_scan(mod, dirs: list[Path]) -> None:
    """把原包的模型扫描换成能跟进目录联接的版本，并锁定模型目录。"""
    mod._llm_roots = lambda: dirs

    def _records(items: list[Path]) -> list[dict]:
        out = []
        for path in items:
            rel = ""
            for root in dirs:
                try:
                    rel = path.relative_to(root).as_posix()
                    break
                except ValueError:
                    continue
            if rel:
                out.append({"name": path.name, "path": str(path), "relative_path": rel})
        return out

    def scan_mmproj() -> list[dict]:
        items = [p for root in dirs if root.is_dir() for p in _walk_gguf(root) if mod._is_mmproj(p)]
        return _records(items)

    def scan_visual() -> list[dict]:
        gguf = [p for root in dirs if root.is_dir() for p in _walk_gguf(root)
                if not mod._is_mmproj(p) and mod._gguf_handler_name(p) is not None]
        by_path = {item["path"]: item["relative_path"] for item in scan_mmproj()}
        result = []
        for item in _records(gguf):
            cands = [by_path[str(c.absolute())] for c in _matching(Path(item["path"]))
                     if str(c.absolute()) in by_path]
            result.append({**item, "format": "gguf", "mmproj_candidates": cands})
        return result

    def _root_of(path: Path) -> Path | None:
        for root in dirs:
            try:
                path.relative_to(root)
                return root
            except ValueError:
                continue
        return None

    def _same_root_first(paths: list[Path], model_path: Path) -> list[Path]:
        """mmproj 优先取与主模型同一个扫描根里的那份，避免跨盘依赖。"""
        root = _root_of(model_path)
        return sorted(
            paths,
            key=lambda p: (0 if _root_of(p) == root else 1, p.name.lower()),
        )

    def _matching(model_path: Path) -> list[Path]:
        projectors = [p for root in dirs if root.is_dir() for p in _walk_gguf(root) if mod._is_mmproj(p)]
        identity = mod._gguf_identity(model_path.name)
        exact = [p for p in projectors if mod._gguf_identity(p.name, projector=True) == identity]
        if exact:
            return _same_root_first(exact, model_path)
        signature = mod._gguf_model_signature(model_path.name)
        compatible = [p for p in projectors if signature and mod._gguf_model_signature(p.name) == signature]
        pool = compatible or projectors
        if not pool:
            return []
        scored = []
        for p in pool:
            cand = mod._gguf_identity(p.name, projector=True)
            from difflib import SequenceMatcher
            containment = 1 if identity in cand or cand in identity else 0
            ratio = SequenceMatcher(None, identity, cand).ratio()
            scored.append(((containment, ratio), p))
        best = max(score for score, _ in scored)
        return _same_root_first([p for score, p in scored if score == best], model_path)

    mod._scan_mmproj_models = scan_mmproj
    mod._scan_visual_models = scan_visual
    mod._matching_mmproj = _matching
    # 纯脚本环境没有 ComfyUI 的模型要卸载
    mod._unload_comfy_models = lambda: None


def load_optimizer(cfg: dict):
    if not TARGET.is_file():
        raise SystemExit(
            "找不到 prompt_optimizer.py。\n"
            f"  期望路径: {TARGET}\n"
            "  请把 Goohai-MiniMax-H3_Integration 整包放进 engine\\vendor。"
        )
    _install_shims()
    spec = importlib.util.spec_from_file_location("_goohai_prompt_optimizer", TARGET)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    dirs = [Path(p) for p in cfg.get("model_dirs") or []]
    _bind_junction_safe_scan(mod, dirs)
    return mod


# ---------------------------------------------------------------- 模型
def pick_model(mod, cfg) -> str:
    models = mod._scan_visual_models()
    gguf = [m for m in models if m.get("format") == "gguf"]
    if not gguf:
        roots = "\n".join(f"  - {p}" for p in cfg.get("model_dirs") or [])
        raise SystemExit(f"没有找到可用的 GGUF 视觉模型，扫描目录：\n{roots}")
    wanted = str(cfg.get("local_model") or "").strip()
    if wanted:
        for m in gguf:
            if m["relative_path"] == wanted or m["path"] == wanted or m["name"] == wanted:
                return m["relative_path"]
        raise SystemExit(f"配置里的 local_model 不存在: {wanted}")
    return gguf[0]["relative_path"]


def print_models(mod, cfg) -> None:
    models = mod._scan_visual_models()
    gguf = [m for m in models if m.get("format") == "gguf"]
    print("可用 GGUF 主模型：")
    for m in gguf:
        size = ""
        try:
            size = f"{Path(m['path']).stat().st_size / 1024 ** 3:.2f} GB"
        except OSError:
            pass
        cand = ", ".join(m.get("mmproj_candidates") or []) or "（无匹配 mmproj）"
        print(f"  · {m['name']}  [{size}]  id={m['relative_path']}\n      mmproj: {cand}")
    if not gguf:
        print("  （无）")
    print("\n扫描目录：")
    for root in cfg.get("model_dirs") or []:
        print(f"  - {root}  {'存在' if Path(root).is_dir() else '不存在/不可访问'}")


# ---------------------------------------------------------------- 图片
def image_to_data_url(path: Path, max_edge: int) -> str:
    from PIL import Image

    img = Image.open(path).convert("RGB")
    img.thumbnail((max_edge, max_edge))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_media(images: list[Path], max_edge: int) -> list[dict]:
    media = []
    for index, path in enumerate(images, start=1):
        if not path.is_file():
            raise SystemExit(f"参考图不存在: {path}")
        media.append({"label": f"<picture {index}>", "kind": "image",
                      "images": [image_to_data_url(path, max_edge)]})
    return media


def build_media_from_spec(spec_path: Path, max_edge: int) -> tuple[list[dict], list[str]]:
    """按节点给出的清单还原 media：图片/视频帧转 data URL，音频只登记标签。"""
    data = json.loads(spec_path.read_text(encoding="utf-8"))
    media: list[dict] = []
    notes: list[str] = []
    for item in data if isinstance(data, list) else []:
        label = str(item.get("label") or "")
        kind = str(item.get("kind") or "image")
        if kind == "audio":
            media.append({"label": label, "kind": "audio"})
            if item.get("note"):
                notes.append(f"{label}：{item['note']}")
            continue
        frames = []
        for path in item.get("images") or []:
            p = Path(path)
            if not p.is_file():
                raise SystemExit(f"参考素材不存在: {p}")
            frames.append(image_to_data_url(p, max_edge))
        if frames:
            media.append({"label": label, "kind": kind, "images": frames})
    return media, notes


# ---------------------------------------------------------------- 剪贴板
def copy_to_clipboard(path: Path) -> bool:
    try:
        import ctypes
        from ctypes import wintypes

        CF_UNICODETEXT, GMEM_MOVEABLE = 13, 0x0002
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
        kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
        kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
        kernel32.GlobalLock.restype = ctypes.c_void_p
        user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
        user32.SetClipboardData.restype = wintypes.HANDLE

        data = path.read_text(encoding="utf-8").encode("utf-16-le")
        handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(data) + 2)
        pointer = kernel32.GlobalLock(handle)
        ctypes.memmove(pointer, data, len(data))
        kernel32.GlobalUnlock(handle)
        if not user32.OpenClipboard(None):
            return False
        try:
            user32.EmptyClipboard()
            user32.SetClipboardData(CF_UNICODETEXT, handle)
        finally:
            user32.CloseClipboard()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[提示] 写剪贴板失败: {exc}")
        return False


# ---------------------------------------------------------------- 主流程
def run(args) -> str:
    cfg = load_config()
    mod = load_optimizer(cfg)

    if args.model:
        cfg["local_model"] = args.model
        cfg["local_mmproj"] = args.mmproj or ""
    elif args.mmproj:
        cfg["local_mmproj"] = args.mmproj

    if args.models:
        print_models(mod, cfg)
        return ""

    prompt = args.prompt or ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).expanduser().read_text(encoding="utf-8")
    if not prompt.strip():
        raise SystemExit("提示词为空。")

    task = (args.task or cfg["task"]).upper().strip()
    task = TASK_ALIASES.get(task, task)
    if task not in TASKS:
        raise SystemExit(f"task 只能是 {', '.join(TASKS)}")
    duration = float(args.duration if args.duration is not None else cfg["duration"])
    language = args.lang or cfg["output_language"]

    images: list[Path] = []
    if args.images:
        parts = [p for p in args.images.replace("，", ",").split(",") if p.strip()]
        images = [Path(p.strip().strip('"')).expanduser() for p in parts]
    max_edge = int(cfg.get("image_max_edge") or 1280)
    notes: list[str] = []
    if args.media_file:
        media, notes = build_media_from_spec(Path(args.media_file).expanduser(), max_edge)
    else:
        media = build_media(images, max_edge)
    if notes:
        prompt = prompt.rstrip() + "\n\n[参考素材信息]\n" + "\n".join(notes)

    node_config = {
        "mode": "local",
        "local_model": pick_model(mod, cfg),
        "local_mmproj": str(cfg.get("local_mmproj") or ""),
        "output_language": language,
        "max_tokens": int(args.max_tokens or cfg["max_tokens"]),
        "read_media": True,
    }
    payload = {
        "prompt": prompt,
        "task": task,
        "duration": duration,
        "media": media,
        "context": {"audio_mode": "native"},
        "config": node_config,
    }

    if args.show_system:
        labels = [item["label"] for item in media]
        print(mod._system_prompt(task, duration, labels, language, payload["context"], prompt))
        return ""

    config, _cancel = mod._prepare_prompt_optimization(payload)

    status = mod._gguf_dependency_status()
    if not status.get("available"):
        raise SystemExit("GGUF 视觉模型依赖不可用: " + json.dumps(status, ensure_ascii=False))

    model = mod._selected_local_model(config)
    mmproj = mod._selected_mmproj(config, model)
    print(f"模型  : {model['path']}")
    print(f"mmproj: {mmproj}")
    print(f"任务  : {task}  时长: {duration:g}s  语言: {language}  参考素材: {len(media)} 项")
    print("生成中，首次加载模型较慢，请稍候…")

    started = _dt.datetime.now()
    result = asyncio.run(mod._run_prompt_optimization(config, payload))
    elapsed = (_dt.datetime.now() - started).total_seconds()

    if not result.strip():
        raise SystemExit("模型返回了空结果。")
    print(f"\n[完成] 用时 {elapsed:.1f}s，输出 {len(result)} 字符\n")

    out_dir = Path(cfg.get("out_dir") or (ROOT / "输出"))
    out_dir.mkdir(parents=True, exist_ok=True)
    target = Path(args.out).expanduser() if args.out else (
        out_dir / f"提示词_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(result, encoding="utf-8")

    print(result)
    print(f"\n已保存: {target}")
    if args.copy and copy_to_clipboard(target):
        print("已复制到剪贴板。")
    return result


# ---------------------------------------------------------------- 交互
def read_prompt_interactive() -> str:
    print("把你的描述粘进来（可以多行）。粘完回车提交，再敲一次回车（空行）即结束。")
    print("-" * 60)
    lines: list[str] = []
    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        s = line.strip()
        if s.upper() in ("EOF", "END", "结束", "完成"):
            break
        if not s:
            if lines:
                break
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="MiniMax H3 提示词优化（独立引擎）")
    p.add_argument("--prompt", help="原始描述")
    p.add_argument("--prompt-file", help="从文件读原始描述（utf-8）")
    p.add_argument("--task", default=None, help="T2VA/I2VA/L2VA/FL2VA/REF2VA/HYBRID")
    p.add_argument("--duration", type=float, default=None, help="目标时长（秒）")
    p.add_argument("--lang", default=None, help="中文 / English")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--model", default=None, help="主模型 id（相对模型目录）")
    p.add_argument("--mmproj", default=None, help="mmproj id")
    p.add_argument("--images", default=None, help="参考图路径，逗号分隔")
    p.add_argument("--media-file", default=None,
                   help="参考素材清单 JSON（图片/视频帧/音频标签，节点用这个）")
    p.add_argument("--out", default=None, help="结果输出路径")
    p.add_argument("--copy", action="store_true", help="把结果复制到剪贴板")
    p.add_argument("--models", action="store_true", help="列出可用模型")
    p.add_argument("--show-system", action="store_true", help="只打印系统提示词")
    p.add_argument("--interactive", action="store_true", help="命令行交互输入描述")
    return p


def main(argv=None) -> int:
    _utf8_streams()
    args = _parser().parse_args(argv)
    if args.interactive and not args.prompt and not args.prompt_file:
        args.prompt = read_prompt_interactive()
    run(args)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        log_dir = ROOT / "日志"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"崩溃_{_dt.datetime.now():%Y%m%d_%H%M%S}.txt"
        import traceback
        path.write_text(traceback.format_exc(), encoding="utf-8")
        print(f"\n[失败] {exc}\n详细日志: {path}")
        sys.exit(1)
