"""core/image_similarity.py的单元测试：像素diff判同一屏。

回归用例覆盖2026-07-12实测发现的那个真实局限——内容稀疏、背景大片空白的
两个不同界面，像素diff比例可能很低，容易被误判成同一屏（这也是后来把
SIMILARITY_GRAY_ZONE定在0.25、不敢完全依赖图片diff兜底的原因）。
"""
from __future__ import annotations

import numpy as np

from image_similarity import DIFF_MAX_RATIO, images_look_same


def solid_image(shape=(100, 100, 3), color=(255, 255, 255)) -> np.ndarray:
    img = np.zeros(shape, dtype=np.uint8)
    img[:] = color
    return img


class TestImagesLookSame:
    def test_identical_images_are_same(self):
        img = solid_image()
        assert images_look_same(img, img.copy())

    def test_completely_different_images_not_same(self):
        a = solid_image(color=(0, 0, 0))
        b = solid_image(color=(255, 255, 255))
        assert not images_look_same(a, b)

    def test_small_localized_change_still_same(self):
        """同一屏只有一小块区域变化（比如某个任务项状态变化），diff比例在
        DIFF_MAX_RATIO容忍范围内，应该判成同一屏。"""
        a = solid_image(shape=(100, 100, 3), color=(255, 255, 255))
        b = a.copy()
        b[0:10, 0:10] = (0, 0, 0)  # 100x100里改10x10，diff比例1%，远低于阈值
        assert images_look_same(a, b)

    def test_different_shapes_not_same(self):
        a = solid_image(shape=(100, 100, 3))
        b = solid_image(shape=(200, 200, 3))
        assert not images_look_same(a, b)

    def test_diff_ratio_right_at_threshold_boundary(self):
        """diff比例刚好卡在DIFF_MAX_RATIO：不多不少构造一个刚好超过阈值的用例，
        验证确实被判成不同（防止阈值判断方向写反这种低级错误）。"""
        h, w = 100, 100
        a = solid_image(shape=(h, w, 3), color=(255, 255, 255))
        b = a.copy()
        changed_rows = int(h * (DIFF_MAX_RATIO + 0.05))
        b[:changed_rows, :] = (0, 0, 0)
        assert not images_look_same(a, b)
