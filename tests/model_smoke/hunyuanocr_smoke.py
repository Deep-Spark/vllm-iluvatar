# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Shanghai Iluvatar CoreX Semiconductor Co., Ltd.
"""Explicit live single-image OCR smoke; does not extend the regular CI suite.

Requires Pillow and an already running OpenAI-compatible model service.
Ten generated documents run serially and then at concurrency four. Answers are
compared in full, ignoring whitespace only; no request is retried. Random-weight
execution checks must use --execution-only and never count as OCR acceptance.
"""

import argparse
import base64
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CASES = [
    "HELLO WORLD\nINVOICE 2026\nTOTAL 128.50",
    "上海天数智芯\n光学字符识别\n测试通过",
    "订单编号 123456\n产品数量 16\n合计 3200",
    "Hunyuan OCR\nModel Validation\nNovember 2026",
    "设备编号 BI-V150\n运行状态 正常\n日期 2026-11-23",
    "0123456789\n9876543210\n1357902468",
    "Meeting Notes\nStart 09:30\nRoom 205",
    "中文识别测试\n第一行文字\n第二行文字",
    "Name Alice\nCity Shanghai\nScore 98",
    "货物清单\n键盘 3\n鼠标 5",
]
PROMPT = "识别图片中的所有文字，逐行输出原文，不要添加解释或 Markdown 标记。"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18123")
    parser.add_argument("--model", default="hunyuanocr-test")
    parser.add_argument(
        "--font", default="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execution-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(args.font, 44)
    images = []
    for index, expected in enumerate(CASES):
        img = Image.new("RGB", (768, 512), "white")
        ImageDraw.Draw(img).multiline_text(
            (40, 60), expected, font=font, fill="black", spacing=30
        )
        image_path = args.output / f"image-{index:02d}.png"
        img.save(image_path)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        images.append(base64.b64encode(buffer.getvalue()).decode())

    def request(index, phase):
        started = time.monotonic()
        result = {"case": index, "phase": phase, "expected": CASES[index]}
        payload = {
            "model": args.model,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "max_tokens": 8 if args.execution_only else 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + images[index]
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
        }
        try:
            req = urllib.request.Request(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as response:
                body = json.load(response)
            result["response"] = body
            choices = body.get("choices", [])
            choice = choices[0] if len(choices) == 1 else {}
            content = choice.get("message", {}).get("content")
            valid = (
                body.get("model") == args.model
                and isinstance(content, str)
                and bool(content.strip())
            )
            result["request_ok"] = valid and choice.get("finish_reason") == "stop"
            result["correct"] = result["request_ok"] and "".join(
                content.split()
            ) == "".join(CASES[index].split())
            result["execution_ok"] = valid and choice.get("finish_reason") in (
                "stop",
                "length",
            )
        except Exception as error:
            result.update(
                error=f"{type(error).__name__}: {error}",
                request_ok=False,
                correct=False,
                execution_ok=False,
            )
        result["seconds"] = round(time.monotonic() - started, 3)
        return result

    results = [
        request(i, "serial") for i in range(1 if args.execution_only else len(CASES))
    ]
    if not args.execution_only:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results.extend(
                pool.map(lambda i: request(i, "concurrent"), range(len(CASES)))
            )
    summary = {
        "mode": "execution_only" if args.execution_only else "ocr_acceptance",
        "total": len(results),
        "request_ok": sum(r["request_ok"] for r in results),
        "correct": sum(r["correct"] for r in results),
        "execution_ok": sum(r["execution_ok"] for r in results),
        "ocr_passed": not args.execution_only
        and len(results) == 20
        and all(r["correct"] for r in results),
    }
    (args.output / "answers.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2)
    )
    (args.output / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, ensure_ascii=False))
    return (
        0
        if (
            all(r["execution_ok"] for r in results)
            if args.execution_only
            else summary["ocr_passed"]
        )
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
