"""比较两张截图像素上是不是"同一屏"

给exploration_memory.py灰色地带（OCR文字相似度判断不出来）的tiebreak用，也给
dedupe_nodes.py离线批量合并重复节点用——两处都需要同一套"允许局部变化、看整体
差异比例"的判断，抽成这一个模块避免各写各的阈值。

不追求逐像素一致：同一屏完全可能有一小块区域在变（比如某个任务项从"可用"变
"冷却中"），只要变化的像素比例不太大就认为是同一屏。阈值是根据一次实测真实
案例估的——明确是同一屏、只有一处子项状态变化，变化像素比例约20%，这里留了
一点余量到35%。
"""
from __future__ import annotations

import numpy as np
import cv2

DIFF_PIXEL_THRESHOLD = 30
DIFF_MAX_RATIO = 0.35


def images_look_same(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    diff = cv2.absdiff(a, b)
    diff_ratio = (diff.max(axis=2) > DIFF_PIXEL_THRESHOLD).mean()
    return diff_ratio <= DIFF_MAX_RATIO
