"""把多張圖片依序串成測試影片，用來在沒有攝影機時模擬作業流程。

用法：
  .venv\\Scripts\\python.exe tools\\make_demo_video.py demo\\demo.mp4 img1.png img2.png ... --seconds 4
  每張圖片可用「路徑@秒數」個別指定秒數，例如 a.png@2.5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--seconds", type=float, default=4.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    rng = np.random.default_rng(0)
    total = 0.0
    for spec in args.images:
        path, _, seconds = spec.partition("@")
        seconds = float(seconds) if seconds else args.seconds
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"讀不到圖片：{path}")
        image = cv2.resize(image, (args.width, args.height), interpolation=cv2.INTER_AREA)
        for _ in range(round(seconds * args.fps)):
            # 加一點雜訊，比較接近真實攝影機畫面
            noise = rng.normal(0, 3, image.shape).astype(np.int16)
            writer.write(np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8))
        total += seconds
        print(f"{Path(path).name}  {seconds:g}s")
    writer.release()
    print(f"已輸出 {output}（{total:g} 秒）")


if __name__ == "__main__":
    main()
