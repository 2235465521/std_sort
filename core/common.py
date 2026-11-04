# -*- coding: utf-8 -*-
"""standard-pdf-classifier / core/common.py
共享骨架：渲染、版心检测、背景估计、批量遍历、追溯清单、日志。

与 standard-pdf-cleaner 的判定判据对齐（像素级、可复现、固定种子）。
本模块不承载清理逻辑，只做"判定 + 批量 + 清单"。
"""
import os
import re
import gc
import sys
import random
import shutil
import logging

import fitz  # PyMuPDF

# ---------------- 日志 ----------------
log = logging.getLogger("stdpdfclassify")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("[classify] %(message)s"))
    log.addHandler(_h)

# ---------------- 类型判定（单一源：core/type_judge.py） ----------------
# 类型判定常量与函数（is_text_pdf / _is_cid_garbled / page_image_coverage / has_large_image / _sample_pages）
# 统一在 core/type_judge.py 定义（判据以 standard-pdf-cleaner 定稿值为准，2026-10-24 用户定稿），
# 本模块与 standard-pdf-cleaner 均从该单一源引用，避免两技能各自维护导致判定漂移。
from core.type_judge import (
    TEXT_MIN_TOTAL_CHARS, TEXT_MIN_PER_PAGE, LARGE_IMAGE_BYTES,
    SAMPLE_MAX_PAGES, _SAMPLE_SEED, TEXT_PER_PAGE_LOW,
    IMAGE_PRESENT_THRESH, IMAGE_COVER_MIXED_THR,
    _CJK_COMMON, _is_cid_garbled,
    page_image_coverage, has_large_image, _sample_pages, is_text_pdf,
)

# 版心检测 / 颜色判定
BODY_GRAY = 150                # 正文暗色灰度阈值（像素级）
HDR_Y_MAX_RATIO = 0.15         # 顶部条带（页眉）比例
FTR_Y_MIN_RATIO = 0.85         # 底部条带（页脚）比例
CHROMA_COLOR = 25              # 彩色水印最小色度（主色-次大通道）
GRAY_CHROMA_MAX = 18           # 灰度水印最大色度（R≈G≈B）
WM_BG_MIN_DIST = 15            # 水印与背景最小差值
WM_LUM_MIN = 40                # 水印亮度下限
WM_LUM_MAX = 245               # 水印亮度上限
GRAY_LUM_MIN = 90              # 灰度水印亮度下限（浅于正文墨）
SAME_DARK_LUM = 150            # 同色水印（暗色）亮度阈值
CROSS_PAGE_MIN = 0.5           # 跨页一致出现比例（>= 一半页）

# 输出目录结构（用户定稿）
TYPE_GARBLED = "乱码型"       # 复制为乱码/CID伪文本
TYPE_TEXT = "文本型"
TYPE_IMAGE = "图像型"
SUB_NO_WM = "无版心水印"
SUB_WM = "有版心水印"
SUB_COLOR = "彩色"
SUB_GRAY = "灰度"
SUB_SAME = "同色"

DEFAULT_INPUT_ROOT = r"E:\STD-Work\01-标准分类\01-分类标准"      # 用户定稿(2026-10-24/25)：用途1 默认输入（源 PDF）；00-标准分类 更正为 01-标准分类
DEFAULT_OUTPUT_ROOT = r"E:\STD-Work\01-标准分类\02-成果输出"     # 用户定稿：默认输出根（02-Pdf分类 层级已取消；00→01 更正）
DEFAULT_P2_INPUT_ROOT = r"E:\STD-Work\01-标准分类\02-成果输出\03-图像型"  # 用途2 默认输入 = 用途1 的 03-图像型，在 03 根内细分
# 用途1（快速，--purpose 1）：只分 文本/乱码/图像型（文本型第一，用户定稿）；04-其他 收容未分类（v1.6.0）
SUB_DIRS_P1 = {
    "text": "01-文本型",
    "garbled": "02-乱码型",
    "image": "03-图像型",
    "other": "04-其他",
}
# 用途2（细分，--purpose 2）：仅"图像型"，在 03-图像型 根目录内再分水印子类（嵌套子目录，v1.6.2 用户定稿命名）
SUB_DIRS_P2 = {
    "image_wm_color": "01-彩色水印",
    "image_wm_gray": "02-灰度水印",
    "image_wm_same": "03-字印同色",
    "image_wm_stripe": "04-污带图像",      # v1.8 新增（2026-10-26 用户定稿）：纸质扫描污带/整面底纹
    "image_nowm": "05-无印图文",
    "image_broken": "06-损坏文件",
}
# 整合用途（v1.7.0 用户定稿 2026-10-25）：单一流程——判定类型，图像型自动细分水印归位
# 输出根 02-成果输出 下：01-文本型 / 02-乱码型 / 03-图像型\{子类} / 04-其他（兜底未分类）
SUB_DIRS = {
    "text": "01-文本型",
    "garbled": "02-乱码型",
    "other": "04-其他",
    "image_wm_color": "03-图像型\\01-彩色水印",
    "image_wm_gray": "03-图像型\\02-灰度水印",
    "image_wm_same": "03-图像型\\03-字印同色",
    "image_wm_stripe": "03-图像型\\04-污带图像",
    "image_nowm": "03-图像型\\05-无印图文",
    "image_broken": "03-图像型\\06-损坏文件",
}
# 简单分类（--simple，用户定稿 2026-10-25）：提示词说"简单分类"时只分 文本/乱码/图像，
# 图像型不细分水印、直接归 03-图像型 根目录（省时）
SUB_DIRS_SIMPLE = {
    "text": "01-文本型",
    "garbled": "02-乱码型",
    "image": "03-图像型",
    "other": "04-其他",
}

MANIFEST_NAME = "classifier_manifest.xlsx"


# ---------------- 渲染 ----------------
def render_rgb(page, zoom=2.0, max_dim=2400):
    """将一页渲染为 RGB ndarray。zoom=2.0 -> 144 DPI（颜色判定足够）。

    max_dim（v1.5.0 内存防护）：渲染后长边上限 2400px——超大物理页面（如标准编号规定类）
    在 zoom=2.0 下会渲染出 4000×6000px（≈72MB/页×2 份），单文件峰值曾实测 1.47GB；
    超限时自动降低 zoom 保住判定精度下限，防批量处理内存溢出。
    返回 (arr HxWx3 uint8, width_pt, height_pt)。
    """
    r = page.rect
    md = max(r.width, r.height)
    z = zoom
    if md > 0 and md * z > max_dim:
        z = max_dim / md
    mat = fitz.Matrix(z, z)
    pm = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)
    import numpy as np
    arr = np.frombuffer(pm.samples, dtype=np.uint8).reshape(pm.height, pm.width, 3)
    return arr.copy(), pm.width, pm.height


# ---------------- 类型判定（单一源已上移至 core/type_judge.py） ----------------
# 判定常量与函数（page_image_coverage/has_large_image/_sample_pages/is_text_pdf/_is_cid_garbled）
# 统一在 core/type_judge.py 定义（判据以 standard-pdf-cleaner 定稿值为准），本模块顶部 import 引用。
# 以下保留"分类视角"辅助函数：把双层（整页大图底 + 文本层）并入图像型。
def classify_text_image(doc):
    """返回 'text' / 'image'（把参考技能的 dual 并入 image：双层有整页大图底，视为图像型）。"""
    if not is_text_pdf(doc):
        return "image"
    # 整页大图底（双层）→ 图像型（分类视角归入图像）
    try:
        for pno in range(doc.page_count):
            page = doc[pno]
            pr = page.rect
            area = pr.width * pr.height
            for img in page.get_images(full=True):
                xref, w, h = img[0], img[2], img[3]
                if w <= 1 or h <= 1:
                    continue
                for r in page.get_image_rects(xref):
                    if r.width * r.height >= area * 0.85:
                        return "image"
    except Exception:
        pass
    return "text"


# ---------------- 版心检测 ----------------
def detect_body_region(arr, cover=0.97):
    """检测页面正文版心 y 范围（暗像素累积主体法，对齐参考技能）。返回 (y_top, y_bottom) 像素行号。"""
    import numpy as np
    h, w = arr.shape[:2]
    gray = arr[:, :, :3].astype(np.int32).mean(axis=2)
    dark = gray < BODY_GRAY
    rowsum = dark.sum(axis=1).astype(np.float64)
    thr = max(2, int(w * 0.0008))
    rows = [(r, rowsum[r]) for r in range(h) if rowsum[r] > thr]
    if not rows:
        return (0, h)
    total = sum(v for _, v in rows)
    order = sorted(rows, key=lambda x: -x[1])
    acc, sel = 0.0, []
    for r, v in order:
        sel.append(r)
        acc += v
        if acc >= total * cover:
            break
    sel = sorted(sel)
    clusters = [[sel[0]]]
    for r in sel[1:]:
        if r - clusters[-1][-1] <= 12:
            clusters[-1].append(r)
        else:
            clusters.append([r])
    best = max(clusters, key=lambda cl: int(sum(rowsum[x] for x in cl)))
    return (best[0], best[-1])


def estimate_bg(arr, region=None):
    """估计背景主色：取区域（默认全图）众数色。返回 (r,g,b)。"""
    import numpy as np
    if region is not None:
        y0, y1 = region
        sub = arr[y0:y1, :, :]
    else:
        sub = arr
    if sub.size == 0:
        return (255, 255, 255)
    q = (sub // 8 * 8 + 4).astype(np.int32)
    key = (q[:, :, 0].astype(np.int64) << 16) | (q[:, :, 1] << 8) | q[:, :, 2]
    vals, counts = np.unique(key, return_counts=True)
    top = int(vals[int(np.argmax(counts))])
    return ((top >> 16) & 255, (top >> 8) & 255, top & 255)


# ---------------- 批量遍历 ----------------
def list_pdfs(root, exclude=None):
    """递归列出目录下所有 .pdf 文件；可排除输出根目录（exclude），避免把已分类副本再当输入。"""
    exclude_abs = os.path.abspath(exclude) if exclude else None
    result = []
    for dirpath, dirnames, filenames in os.walk(root):
        dp = os.path.abspath(dirpath)
        if exclude_abs and (dp == exclude_abs or dp.startswith(exclude_abs + os.sep)):
            dirnames[:] = []
            continue
        if exclude_abs:
            dirnames[:] = [d for d in dirnames
                           if os.path.join(dp, d) != exclude_abs
                           and not os.path.join(dp, d).startswith(exclude_abs + os.sep)]
        for fn in filenames:
            if fn.lower().endswith(".pdf"):
                result.append(os.path.join(dirpath, fn))
    return sorted(result)


def list_pdfs_top(root):
    """仅列输入根目录【顶层】PDF（用途2：只细分 03-图像型 根目录下文件，不递归子目录，避免重复细分）。"""
    result = []
    try:
        for fn in os.listdir(root):
            if fn.lower().endswith(".pdf"):
                result.append(os.path.join(root, fn))
    except OSError:
        pass
    return sorted(result)


def open_doc(path):
    """打开 PDF。加密需密码返回 None（调用方判未分类，不归乱码型）。返回 (doc 或 None, 是否需密码)。"""
    try:
        doc = fitz.open(path)
        if doc.needs_pass:
            # 尝试空密码
            try:
                if not doc.authenticate(""):
                    doc.close()
                    return None, True
            except Exception:
                doc.close()
                return None, True
        return doc, False
    except Exception:
        return None, False


def copy_to(pdf_path, dest_subdir_abs, manifest_row, dest_name=None):
    """复制源 PDF 到目标子目录。返回 (目标路径, ok, skipped)。

    dest_name（v1.7.0）：指定目标文件名（带类型后缀，如 `_文本`/`_乱码`/`_彩色`/`_图像`），
    不传则用源文件名。续跑去重（v1.4.0）：目标已存在且与源文件大小相同 → 视为已归位，
    跳过不重复复制（先 200 试跑、后全量衔接时，全量跳过试跑已归位文件，不产生 _1 重复副本）。
    """
    os.makedirs(dest_subdir_abs, exist_ok=True)
    name = dest_name or os.path.basename(pdf_path)
    dest = os.path.join(dest_subdir_abs, name)
    if os.path.abspath(dest) == os.path.abspath(pdf_path):
        return dest, True, True  # 已在目标，跳过复制
    if os.path.exists(dest):
        try:
            if os.path.getsize(dest) == os.path.getsize(pdf_path):
                return dest, True, True  # 同源已归位（续跑跳过）
        except OSError:
            pass
        base, ext = os.path.splitext(name)
        i = 1
        while os.path.exists(os.path.join(dest_subdir_abs, f"{base}_{i}{ext}")):
            i += 1
        dest = os.path.join(dest_subdir_abs, f"{base}_{i}{ext}")
    try:
        shutil.copy2(pdf_path, dest)
        return dest, True, False
    except Exception as e:
        manifest_row["状态"] = f"复制失败:{e}"
        return None, False, False


def move_to(pdf_path, dest_subdir_abs, manifest_row):
    """移动 PDF 到目标子目录（用途2 细分归位：从 03-图像型 根目录移入子类子目录）。返回 (目标路径, ok, skipped)。

    续跑去重（v1.6.4，对齐 copy_to）：目标已存在且与源文件大小相同 → 视为已归位跳过，不产生 _1 重复副本
    （用途2 中断重跑 / 重复细分时幂等）。
    """
    os.makedirs(dest_subdir_abs, exist_ok=True)
    name = os.path.basename(pdf_path)
    dest = os.path.join(dest_subdir_abs, name)
    if os.path.abspath(dest) == os.path.abspath(pdf_path):
        return dest, True, True  # 已在目标，跳过移动
    if os.path.exists(dest):
        try:
            if os.path.getsize(dest) == os.path.getsize(pdf_path):
                return dest, True, True  # 同源已归位（续跑跳过）
        except OSError:
            pass
        base, ext = os.path.splitext(name)
        i = 1
        while os.path.exists(os.path.join(dest_subdir_abs, f"{base}_{i}{ext}")):
            i += 1
        dest = os.path.join(dest_subdir_abs, f"{base}_{i}{ext}")
    try:
        shutil.move(pdf_path, dest)
        return dest, True, False
    except Exception as e:
        manifest_row["状态"] = f"移动失败:{e}"
        return None, False, False
