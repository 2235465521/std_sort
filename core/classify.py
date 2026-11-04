# -*- coding: utf-8 -*-
"""standard-pdf-classifier / core/classify.py
核心分类判定：
  1) 类型：乱码型 / 文本型 / 图像型（乱码型=复制为乱码/CID伪文本）
  2) 图像型 -> 有无版心水印
  3) 有版心水印 -> 彩色 / 灰度 / 同色

所有颜色/可量化属性一律【像素级】统计判定（KMeans/直方图/聚类），
不依赖视觉模型的文字描述（对齐用户强规则）。
"""
import os
import gc

import numpy as np

from core import common as C

# 复用 common 常量
BODY_GRAY = C.BODY_GRAY
CHROMA_COLOR = C.CHROMA_COLOR
GRAY_CHROMA_MAX = C.GRAY_CHROMA_MAX
WM_BG_MIN_DIST = C.WM_BG_MIN_DIST
WM_LUM_MIN = C.WM_LUM_MIN
WM_LUM_MAX = C.WM_LUM_MAX
GRAY_LUM_MIN = C.GRAY_LUM_MIN
SAME_DARK_LUM = C.SAME_DARK_LUM
CROSS_PAGE_MIN = C.CROSS_PAGE_MIN


# ---------------- 版心水印检测 ----------------
def _body_wm_mask(arr, bg):
    """在版心内检测"非背景、非暗色正文"的水印候选掩膜（排除页眉/页码条带）。

    策略（对齐参考技能 auto_detect_wm_color）：
      - 只统计主体区（HDR~FTR 条带之间）内非背景像素；
      - 排除暗色正文墨（灰度 < BODY_GRAY）——正文是内容，不是水印；
      - 剩余像素即"版心内非正文非背景"内容 = 水印/印章/图章候选。
    返回布尔掩膜（True=水印候选），以及对应主体区。
    """
    H, W = arr.shape[:2]
    y0 = int(H * C.HDR_Y_MAX_RATIO)
    y1 = int(H * C.FTR_Y_MIN_RATIO)
    if y1 <= y0:
        y0, y1 = 0, H
    main = arr[y0:y1, :, :]
    bg = np.array(bg, dtype=np.int32)
    q = (main // 8 * 8 + 4).astype(np.int32)
    diff = np.abs(q - bg).sum(axis=2)
    gray = q.mean(axis=2)
    dark_body = gray < BODY_GRAY       # 暗色正文墨
    wm = (diff > WM_BG_MIN_DIST) & (~dark_body)
    return wm, (y0, y1)


def _estimate_ink(arr, body_region):
    """估计正文墨色（版心内暗像素低分位）→ 亮度标量，用于"同色水印"相对判定。

    v1.7.1 修正：由"暗像素众数"改为 **P15 低分位**——大面积暗色水印本体（双圆环 logo
    lum 60~65）会拉高众数（实测腌猪肉众数=60，真文字墨=46），低分位取整页最暗端
    （正文文字墨），避免水印本体污染正文墨参照。
    """
    y0, y1 = body_region
    sub = arr[y0:y1, :, :]
    if sub.size == 0:
        return 0.0
    gray = sub.astype(np.int32).mean(axis=2)
    dark = gray < SAME_DARK_LUM
    vals = gray[dark]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, 15))


def _analyze_wm_color(wm_pixels, ink_lum=None):
    """对版心内水印像素做颜色子类判定：彩色 / 灰度 / 同色。

    像素级判据：
      chroma = max(R,G,B) - min(R,G,B)；lum = (R+G+B)/3
      彩色：chroma >= CHROMA_COLOR（有明确色相）；
      灰度：R≈G≈B 且为【浅色】（lum >= GRAY_LUM_MIN），独立浅灰水印；
      同色：R≈G≈B 且为【暗色】/【与正文墨亮度接近】——水印与正文同色系，
           颜色判据失效，归"同色"。
    返回 'color' / 'gray' / 'same' / None。
    """
    if wm_pixels is None or len(wm_pixels) == 0:
        return None
    px = np.array(wm_pixels, dtype=np.int32)
    r, g, b = px[:, 0], px[:, 1], px[:, 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    chroma = mx - mn
    lum = (r + g + b) / 3.0
    valid = (lum >= WM_LUM_MIN) & (lum <= WM_LUM_MAX)
    color = valid & (chroma >= CHROMA_COLOR)
    same = valid & (chroma <= GRAY_CHROMA_MAX) & (lum < GRAY_LUM_MIN)   # 暗色灰=与正文同色
    gray = valid & (chroma <= GRAY_CHROMA_MAX) & (lum >= GRAY_LUM_MIN)  # 浅灰=独立灰度水印
    # 若给出正文墨亮度参照，优先用"接近正文墨"判同色（相对判据）
    if ink_lum is not None:
        close = np.abs(lum - ink_lum) <= 45
        # v1.6.6 修复：同色必须【绝对暗色 lum<GRAY_LUM_MIN】+ 接近正文墨；
        # 浅灰水印（lum>=90）即使页面正文墨参照偏亮（如内容稀疏页 ink≈140）也不判同色
        # （实测 KA_T_1120-2023 几何水印为浅灰，旧逻辑因 close 误判 same=6/10）
        same = valid & (chroma <= GRAY_CHROMA_MAX) & (lum < GRAY_LUM_MIN) & close
        gray = valid & (chroma <= GRAY_CHROMA_MAX) & (~same) & (lum >= GRAY_LUM_MIN)
    # v1.8（用户判据 2026-10-26）：彩色【存在即归彩色】——彩色像素占比 ≥1%（且≥4点）直接返回
    # color，不再按占比最大类（修复页内灰色扫描残留多时彩色被淹没：GB_31657.6 彩色占 wm 37%
    # 仍被 gray 63% 压掉、GB_2763.1 彩色 2.4% 判 gray 的误判）。
    # v1.8 修订（像素级实测校准）：加【平均 chroma ≥50】闸门排除"浅灰抗锯齿伪彩色"——
    # SJ_T_11693.1 浅灰圆环边缘 chroma 35~45（占比 15~36%）误判彩色；真彩色水印
    # （GB_31657.6 彩色水印）平均 chroma 107~113、红章类（GB_2763.1 p20）55；
    # 伪彩色平均 chroma <50 排除，回归灰度本体裁决。
    n = len(px)
    if color.sum() >= max(4, n * 0.01):
        if float(chroma[color].mean()) >= 50.0:
            return "color"
    # 取占比最大的明确类
    cnt = {"color": int(color.sum()), "same": int(same.sum()), "gray": int(gray.sum())}
    best = max(cnt.items(), key=lambda kv: kv[1])
    if best[1] >= max(4, n * 0.10):
        return best[0]
    return None


def _cross_page_same_color(acc, n_pages, H, W):
    """跨页一致暗色水印（同色水印）检测：与正文同色（暗色）且固定位置跨页重复。

    入参为【已逐页累加】的暗墨位置计数 acc（v1.5.0 内存优化：不再保留全部页渲染数组，
    调用方在逐页渲染时同步累加 int32 计数，此处只做阈值+连通域，峰值内存大幅下降）。
    判据：
      - 暗墨位置出现 >= 一半页（CROSS_PAGE_MIN）即候选；
      - 用【连通域块状判定】排除细长文本行：候选暗区须形成有一定面积的矩形连通块
        （文本行是细长形，不构成块；水印/印章/logo 是块状）——避免把正文误判同色。
    返回 True（存在同色水印）/ False。
    """
    import cv2
    half = max(1, int(n_pages * CROSS_PAGE_MIN))
    consistent = (acc >= half).astype(np.uint8)
    # 形态学闭合，合并断裂块
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, kernel)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if nlab <= 1:
        return False
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        # 块状判据：排除"极细长条"（文本行/分隔线：某一维 <15px 且长宽比悬殊），
        # 保留有一定高度/宽度的块（水印/印章/logo 是块状，哪怕较扁如 banner 水印）
        if area < 30:
            continue
        thin = (h < 15 and w > h * 15) or (w < 15 and h > w * 15)
        if thin:
            continue
        # 跨页一致暗块存在
        return True
    return False


# ---------------- 水印本体定位 + 本体墨色裁决（v1.7.1 完善，用户判据） ----------------
# 用户判据（2026-10-26 定稿）：必须先【定位水印本体】（跨页同位稳定出现的块状暗区），
# 测【本体自身墨色】再裁决同色/灰度——不能用整页"非背景非正文"残留像素聚类（浅灰扫描噪点/
# 水印边缘会把本体暗墨淹没，双圆环 logo 误判灰度，如 SB_T_10294-2012 腌猪肉实测）。
# 定位基于【像素级跨页一致计数 acc】（文字行各页位置不同、不累计；水印/logo 跨页同位、
# 累计 ≥ 一半页）——8×8 粗网格会误把"文字普遍存在的版面带"当跨页一致（SB_T_10729 误报）。
def _wm_body_ink_from_acc(acc, acc_lum, n_pages, ink_lum, acc_chroma=None):
    """跨页一致暗块（水印本体）→ 本体墨色 → 同色/灰度裁决（v1.7.1）；本体彩色检测（v1.8.1）。

    入参：
      acc       int32 版心带暗位置跨页计数（逐页累加）；
      acc_lum   int32 同位置 lum 值累加（本体墨色回读用）；
      acc_chroma int32 同位置 chroma 值累加（v1.8.1 本体彩色检测）；
      ink_lum   正文墨亮度参照（_estimate_ink P15 分位）。
    判据：
      - 一致位置 = acc >= max(1, n_pages*CROSS_PAGE_MIN)；
      - 块状连通域 + 暗像素密度 ≤ 0.35（空心/稀疏结构=水印图案；文字密集区密度高排除）；
      - 【v1.8.1 本体彩色】一致块内 chroma≥25 像素占比≥50% 且平均 chroma≥40 → 'color'
        （真彩色水印本体整块有色相：SH_T_1824 墨绿 chroma 51、SJ 棕褐 47~57；
        同色双圆环灰黑 chroma≈10 不触发，腌猪肉案回归 same 不受影响）；
      - 本体墨色 = 一致块内各位置 acc_lum/acc 的 P25（暗端）；
      - 本体墨色 lum < GRAY_LUM_MIN(90) 且 |lum - 正文墨lum| ≤ 45 → 'same'；
      - 本体墨色 lum ≥ GRAY_LUM_MIN → 'gray'；不可靠 → None。
    """
    import cv2
    if acc is None or acc_lum is None:
        return None, "无跨页一致计数"
    if ink_lum < 30 or ink_lum > 190:
        return None, f"正文墨参照异常(ink={ink_lum:.1f})，本体裁决不可靠（深色/超亮页），回退投票"
    H, W = acc.shape
    half = max(1, int(n_pages * CROSS_PAGE_MIN))
    consistent = (acc >= half).astype(np.uint8)
    if not consistent.any():
        return None, "无跨页一致暗块（本体未定位）"
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, kernel)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if nlab <= 1:
        return None, "无跨页一致暗块（本体未定位）"
    lum_avg = np.where(acc > 0, acc_lum.astype(np.float64) / np.maximum(acc, 1), 255.0)
    chroma_avg = (np.where(acc > 0, acc_chroma.astype(np.float64) / np.maximum(acc, 1), 0.0)
                  if acc_chroma is not None else None)
    cand_lums = []
    cand_chromas = []
    cand_area = 0
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 30:
            continue
        if w > h * 8 or h > w * 8:
            continue
        density = area / float(w * h) if w * h else 1.0
        if density > 0.35:      # 空心/稀疏结构（水印图案）；文字密集区排除
            continue
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        sub = lum_avg[y:y + h, x:x + w]
        sub_acc = acc[y:y + h, x:x + w]
        valid = sub_acc > 0
        vals = sub[valid]
        if vals.size < 20:
            continue
        cand_lums.append(vals)
        cand_area = max(cand_area, area)   # v1.8.1：单块最大（文字散碎字符块累加会误过500）
        if chroma_avg is not None:
            sub_ch = chroma_avg[y:y + h, x:x + w]
            cand_chromas.append(sub_ch[valid])
    if not cand_lums:
        return None, "跨页一致暗块均被形态/密度过滤（本体非空心图案）"
    # v1.7.1 本体体量下限：水印图案（双圆环/logo）跨页一致单块 >=500px；
    # 页面同位置散碎字迹碎屑不构成水印本体（SB_T_10729 最大合格块 <300）
    if cand_area < 500:
        return None, f"跨页一致暗块最大合格块={cand_area}<500，非水印本体，回退投票"
    # v1.8.1 本体彩色优先裁决：本体块内 chroma≥25 占比≥50% 且平均 chroma≥40 → 彩色水印
    # （彩色水印本体被 _body_wm_mask 当暗色正文墨排除，wm 候选无彩色；本体通道补上）
    if cand_chromas:
        allch = np.concatenate(cand_chromas)
        ch_ratio = float((allch >= CHROMA_COLOR).mean())
        ch_mean = float(allch.mean())
        if ch_ratio >= 0.5 and ch_mean >= 40.0:
            return "color", (f"本体墨色lum={float(np.percentile(np.concatenate(cand_lums),25)):.1f} "
                             f"本体chroma平均={ch_mean:.0f}(≥25占比{ch_ratio:.0%}) → 彩色（本体有色相，"
                             f"被正文墨阈值排除于wm候选）")
    allv = np.concatenate(cand_lums)
    p25 = float(np.percentile(allv, 25))
    darker = float((allv <= p25 - 40).mean()) if allv.size else 0.0
    detail = (f"本体墨色lum={p25:.1f} 正文墨lum={ink_lum:.1f} Δlum={abs(p25 - ink_lum):.1f} "
              f"本体穿透占比={darker:.3f}")
    if p25 < GRAY_LUM_MIN and abs(p25 - ink_lum) <= 45:
        return "same", detail + " → 同色（本体暗色且接近正文墨，文字被遮掩）"
    if p25 >= GRAY_LUM_MIN:
        return "gray", detail + " → 灰度（本体为独立浅灰，文字可穿透）"
    return None, detail


def _body_max_block_area(acc, n_pages):
    """跨页一致暗块中【最大合格连通块面积】（v1.8.1 新增）。

    合格块 = 跨页一致(acc>=半页) + 块状(宽高比≤8) + 空心/稀疏(密度≤0.35) + 面积≥30。
    用途：替代"跨页一致像素总数"判定水印存在性——密排正文扫描件每页正文在固定
    版面网格，大量字符坐标 2/4 页重合（SB_T_10729 acc>=2 达 6.5 万像素，无水印）；
    真水印本体是【单个连贯大图案块】（腌猪肉圆环 3343、SJ 棕褐 5912）。文字散碎
    字符块单个 <300px，真水印单块 >=500px。
    """
    import cv2
    if acc is None:
        return 0
    half = max(1, int(n_pages * CROSS_PAGE_MIN))
    consistent = (acc >= half).astype(np.uint8)
    if not consistent.any():
        return 0
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, kernel)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best = 0
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 30:
            continue
        if w > h * 8 or h > w * 8:
            continue
        density = area / float(w * h) if w * h else 1.0
        if density > 0.35:
            continue
        best = max(best, area)
    return best


def _wm_logo_max_block(wm_acc, n_pages):
    """浅灰 logo 水印通道：wm_mask（非背景非暗色）跨页一致累加中【近方形大图案块】最大面积。

    用途（v1.8.1）：浅灰 logo 水印（lum>150，文字可穿透圆环/S形）不进暗色 acc 通道
    （acc 只抓 gray<SAME_DARK_LUM=150），导致 AQ_7018 双圆环/SN_T_3701.13 S形 logo
    误判无印。本通道逐页把 _body_wm_mask 结果跨页累加，近方形（宽高比≤4）大图案块
    = logo 本体。区分真无印：GB_3053 最大一致碎块 280、SB_T_10729 4550 块为细长文字行
    （宽高比 18:1，被近方形过滤）。阈值 5000：真 logo 块 2.3万~7.4万。
    """
    import cv2
    if wm_acc is None:
        return 0
    half = max(1, int(n_pages * CROSS_PAGE_MIN))
    consistent = (wm_acc >= half).astype(np.uint8)
    if not consistent.any():
        return 0
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, kernel)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best = 0
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 5000:
            continue
        if w > h * 4 or h > w * 4:   # 近方形 logo；细长文字行（宽高比>4）排除
            continue
        best = max(best, area)
    return best


def _is_bg_layer(consistent, H, W):
    """整面扫描底纹/污带判定（v1.8，用户定稿新增"污带图像"类型）。

    判据（像素级实测校准，2026-10-26）：
      - 跨页一致的残留若为"整面平滑灰底"（扫描底纹/污带）：覆盖率 > 10% 版心，
        且存在面积 > 版心 5% 的巨型"空心/稀疏"连通块；
      - 真水印图案（腌猪肉/SJ/logo）：最大低密度块 1,543~4,342px（图案尺寸），
        远小于版心 5%（~30,000px），覆盖率 ≤10%。
    用户定性样本：CY_T_104.5（扫描污带）一致残留遍布版心（行2%-71%×列1%-99%），
    最大低密度块 90,819px 占版心 ~55%；其渲染无图案形态（整面灰底）。
    """
    if consistent is None or not consistent.any():
        return False
    coverage = float(consistent.sum()) / float(max(1, H * W))
    if coverage <= 0.10:
        return False
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(consistent, cv2.MORPH_CLOSE, kernel)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    max_area = 0
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < 30:
            continue
        density = area / float(w * h) if w * h else 1.0
        if density > 0.35:      # 只统计"空心/稀疏"块（图案/底纹碎片；实心文字密集区不算）
            continue
        max_area = max(max_area, area)
    return max_area > 0.05 * (H * W)


# ---------------- 图像型：版心水印判定 ----------------
def image_wm_verdict(doc):
    """对图像型 PDF 判定版心水印情况。

    返回 dict:
      {has_wm: bool, subtype: 'color'|'gray'|'same'|None, detail: str}

    内存设计（v1.5.0 重构）：**逐页增量处理**——每页渲染后立即提取
    水印计数/颜色投票/暗墨位置累加，随即释放该页大数组，不再保留全部候选页的
    渲染数组（旧实现 arrs+wm_masks+body_regions 全量保留，超大页面单文件峰值
    实测 1.47GB；重构后峰值降至单页量级）。
    """
    cand_pages = C._sample_pages(doc)
    if not cand_pages:
        cand_pages = list(range(min(doc.page_count, C.SAMPLE_MAX_PAGES)))
    # 跨页一致检测需要足够页数：抽样不足 3 页时扩充到全部页（图像型页数通常可控）
    if len(cand_pages) < 3 and doc.page_count > len(cand_pages):
        cand_pages = list(range(doc.page_count))
    wm_counts = []          # 每页水印候选像素数（标量，不保留掩膜）
    wm_color_votes = []     # 每页的颜色子类投票
    ink_estimates = []      # 每页正文墨亮度（本体墨色裁决的相对参照，跨页取中位）
    page_color_full = []     # v1.9 全页彩色通道：每页 (彩色像素占比, 平均chroma) 不限明暗
    n_rendered = 0
    acc = None              # 跨页同色增量：暗墨位置计数 int32（逐页累加，像素级跨页一致）
    acc_lum = None          # 跨页同色增量：同位置 lum 累加（本体墨色回读，v1.7.1）
    acc_chroma = None       # 跨页同色增量：同位置 chroma 累加（本体彩色检测，v1.8.1）
    wm_acc = None           # 浅灰 logo 通道：wm_mask（非背景非暗色）跨页一致累加（v1.8.1）
    base_h = base_w = 0
    for i in cand_pages:
        try:
            arr, _w, _h = C.render_rgb(doc[i], zoom=2.0, max_dim=2400)
        except Exception:
            continue
        # 页级整体异常保护（v1.6.4）：渲染成功但后续掩膜/背景/投票任一步异常 → 跳过该页，
        # 不中断整文件判定（避免误进"判定失败"）；全部页失败才走 broken。
        try:
            if acc is None:
                base_h, base_w = arr.shape[0], arr.shape[1]
                acc = np.zeros((base_h, base_w), dtype=np.int32)
                acc_lum = np.zeros((base_h, base_w), dtype=np.int32)
                acc_chroma = np.zeros((base_h, base_w), dtype=np.int32)
            n_rendered += 1
            # 跨页同色通道：主体带内暗墨位置累加（页尺寸不一致时 resize 到基准页，避免广播失败）
            px = arr[:, :, :3].astype(np.int32)
            gray = px.mean(axis=2)
            chroma = px.max(axis=2) - px.min(axis=2)
            dark = gray < SAME_DARK_LUM
            try:
                by0, by1 = C.detect_body_region(arr)
            except Exception:
                by0, by1 = 0, arr.shape[0]
            mask = np.zeros_like(dark)
            mask[by0:by1, :] = dark[by0:by1, :]
            lum_acc = np.where(mask, gray, 0).astype(np.int32)   # gray 为 float64（np.mean），必须显式 int32
            chm_acc = np.where(mask, chroma, 0).astype(np.int32)
            if mask.shape != (base_h, base_w):
                import cv2
                mask = cv2.resize(mask.astype(np.uint8), (base_w, base_h)) > 0
                lum_acc = cv2.resize(lum_acc.astype(np.float32), (base_w, base_h)).astype(np.int32)
                chm_acc = cv2.resize(chm_acc.astype(np.float32), (base_w, base_h)).astype(np.int32)
            acc += mask.astype(np.int32)
            acc_lum += lum_acc
            acc_chroma += chm_acc
            # 背景：在【主体大区带】内取众数（避免版心窄带/页眉页码污染）
            band = (int(arr.shape[0] * C.HDR_Y_MAX_RATIO), int(arr.shape[0] * C.FTR_Y_MIN_RATIO))
            bg = C.estimate_bg(arr, band)
            wm, (wy0, wy1) = _body_wm_mask(arr, bg)
            # v1.8.1 浅灰 logo 通道：wm_mask 全页坐标跨页累加（lum>150 浅灰 logo 不进暗色 acc）
            wm_full = np.zeros(arr.shape[:2], dtype=np.int32)
            wm_full[wy0:wy1, :] = wm.astype(np.int32)
            if wm_acc is None:
                wm_acc = np.zeros((base_h, base_w), dtype=np.int32)
            if wm_full.shape != (base_h, base_w):
                wm_full = cv2.resize(wm_full.astype(np.uint8), (base_w, base_h)).astype(np.int32)
            wm_acc += wm_full
            wm_counts.append(int(wm.sum()))
            # v1.9 全页彩色通道：不限明暗，**版心带(0.15~0.85)内**非背景像素(diff>15)
            # 中 chroma>=25 的占比与均值。页眉页脚固定彩色元素（SN_T_1772 每页恒定 8817
            # 像素页眉彩色）排除在版心外。深色相彩色水印（墨绿/棕褐 lum<150）不进 wm_mask，
            # 旧窄带 acc 漏判；版心带内全页统计不依赖 detect_body_region。
            bg_arr = np.array(bg, dtype=np.int32)
            diff = np.abs(px - bg_arr).max(axis=2)
            nonbg = np.zeros(diff.shape, dtype=bool)
            nonbg[band[0]:band[1], :] = (diff[band[0]:band[1], :] > 15)
            n_nonbg = int(nonbg.sum())
            if n_nonbg > 0:
                ch_nonbg = chroma[nonbg]
                n_color = int((ch_nonbg >= 25).sum())
                ratio = n_color / float(n_nonbg)
                avg_ch = float(ch_nonbg[ch_nonbg >= 25].mean()) if n_color > 0 else 0.0
                page_color_full.append((ratio, avg_ch))
            # 正文墨亮度参照（用于同色水印相对判定）
            ink = _estimate_ink(arr, (wy0, wy1))
            if ink > 0:
                ink_estimates.append(ink)
            # 提取水印像素样本（注意 wm 是主体带切片后数组，yidx 需加 wy0 偏移回全图坐标）
            yidx, xidx = np.where(wm)
            if len(yidx):
                full_y = wy0 + yidx
                step = max(1, len(full_y) // 4000)
                samp = arr[full_y[::step], xidx[::step], :]
                sub = _analyze_wm_color(samp.tolist(), ink)
                if sub:
                    wm_color_votes.append(sub)
            # 释放本页大数组（渲染图/灰度/暗掩膜/水印掩膜）
            del arr, px, gray, chroma, dark, mask, lum_acc, chm_acc, wm, yidx, xidx, wm_full
        except Exception:
            try:
                del arr
            except Exception:
                pass
            continue
    if not wm_counts:
        return {"has_wm": False, "subtype": None, "detail": "无有效页可渲染（疑似损坏，归 05-损坏文件）",
                "broken": True}

    # 1) 版心内是否有"非背景非正文"内容（水印存在性，v1.8 升级：真水印 = 跨页同位图案
    #    [acc 一致 ≥500] 或 彩色像素；纯随机扫描噪点/底纹无跨页一致 → 无印图文。
    #    修复 GB_3053/GB_T_7781/8162/25336/30055/36540/36908 等 7 个无印文件
    #    （非背景残留 20~64 万/页但跨页一致 0~135）误判灰度）
    total_wm = sum(wm_counts)
    half_n = max(1, int(n_rendered * CROSS_PAGE_MIN))
    # v1.8.1：跨页一致体量用【最大合格图案块面积】（密排正文文字网格重合像素总数
    # 可达 6.5 万但单块 <300px；真水印本体单块 >=500px）
    cons_area = _body_max_block_area(acc, n_rendered)
    color_n = wm_color_votes.count("color")

    # v1.9 优先级①：全页彩色（不限明暗、不依赖 detect_body_region 窄带）——
    # 深色相彩色水印（墨绿/棕褐 lum<150）不进 wm_mask，旧窄带 acc 漏判。
    # 真彩色水印/红章全页必命中（GB_31657.6 chroma107、GB_2763.1 红章55）；
    # 浅灰抗锯齿伪彩色平均chroma<50 排除。
    color_full_pages = [s for s in page_color_full if s[0] >= 0.01 and s[1] >= 50]
    if color_full_pages:
        best = max(color_full_pages, key=lambda s: s[1])
        return {"has_wm": True, "subtype": "color",
                "detail": f"全页彩色水印（{len(color_full_pages)}/{len(page_color_full)} 页彩色占比≥1%、平均chroma={best[1]:.0f}≥50，存在即归彩色）"}

    # v1.9 优先级②：浅灰 logo（wm_mask 跨页一致近方形大块，文字可穿透圆环/S形）
    # 阈值 20000（v1.9.1）：真 logo AQ_7018=74356/SN_T_3701.13=23455；
    # 真无印小图标 JB_T_14236=9601 排除；整面底纹 SY_T_0087.1=869814 走污带前置
    wm_logo = _wm_logo_max_block(wm_acc, n_rendered)
    if wm_logo >= 20000:
        # v1.9 污带前置：wm_logo 通道触发前先查整面底纹/污带（覆盖率>10%）
        wm_cons = (wm_acc >= half_n).astype(np.uint8)
        wm_cov = float(wm_cons.sum()) / float(max(1, base_h * base_w))
        if wm_cov > 0.10:
            return {"has_wm": True, "subtype": "stripe",
                    "detail": f"wm跨页一致整面扫描底纹/污带（覆盖率 {wm_cov:.1%}>10%）→ 污带图像"}
        # 浅灰 logo 通道触发时先查本体 chroma（深色相 logo 墨绿/棕褐本体暗被 wm_mask 排除）
        ink_med = float(np.median(ink_estimates)) if ink_estimates else 0
        body_v, body_detail = _wm_body_ink_from_acc(acc, acc_lum, n_rendered, ink_med, acc_chroma)
        if body_v == "color":
            return {"has_wm": True, "subtype": "color",
                    "detail": f"wm_logo块={wm_logo} + 本体彩色：{body_detail}"}
        return {"has_wm": True, "subtype": "gray",
                "detail": f"浅灰logo水印（wm跨页一致近方形块={wm_logo}≥20000，文字可穿透，归灰度）"}

    # v1.9 优先级③：暗色 acc 最大合格块≥500 → 有水印，走颜色细分（污带/本体裁决/投票）
    if cons_area < 500:
        return {"has_wm": False, "subtype": None,
                "detail": f"版心无非背景内容（跨页一致合格块 {cons_area}<500、wm-logo {wm_logo}<20000 且全页无彩色页：浅灰扫描噪点/底纹归无印图文）"}

    # 2) 颜色子类：彩色 / 灰度 / 同色
    gray_n = wm_color_votes.count("gray")
    same_n = wm_color_votes.count("same")
    total_vote = len(wm_color_votes)
    # 用户定稿（2026-10-22 + v1.8 2026-10-26）：存在明确彩色水印页即归【彩色】，不按多数投票
    # （v1.8 将 _analyze_wm_color 彩色阈值降为 ≥1% 存在即彩色，修复 GB_31657.6 彩色占 wm 25.5%
    # 仍被页内灰色残留 63% 淹没、GB_2763.1 彩色 2.4% 判灰度的漏判）
    if color_n > 0:
        return {"has_wm": True, "subtype": "color",
                "detail": f"彩色水印（{color_n}/{total_vote} 页检出彩色像素，存在即归彩色）"}
    # v1.8 底纹排除：跨页一致但为整面扫描底纹/污带（覆盖率>10% 且巨型低密度块>版心5%）
    # → 新增"污带图像"子类（用户定稿，CY_T_104.5：纸质扫描污带非水印）
    cons = (acc >= half_n).astype(np.uint8)
    if _is_bg_layer(cons, base_h, base_w):
        cov = float(cons.sum()) / float(max(1, base_h * base_w))
        return {"has_wm": True, "subtype": "stripe",
                "detail": f"跨页一致整面扫描底纹/污带（覆盖率 {cov:.1%}、巨型低密度块>版心5%），"
                          f"无图案形态 → 污带图像（非水印）"}
    # v1.7.1 本体墨色裁决（用户判据 2026-10-26：先定位水印本体，测本体自身墨色，
    # 再裁决同色/灰度——修复 SB_T_10294-2012 腌猪肉类双圆环暗墨本体被 _body_wm_mask
    # 当正文墨排除、只剩浅灰边缘投票而误判灰度的问题）
    ink_med = float(np.median(ink_estimates)) if ink_estimates else 0.0
    body_v, body_detail = _wm_body_ink_from_acc(acc, acc_lum, n_rendered, ink_med, acc_chroma)
    if body_v == "color":
        return {"has_wm": True, "subtype": "color", "detail": f"跨页一致水印本体（彩色）：{body_detail}"}
    if body_v == "same":
        return {"has_wm": True, "subtype": "same", "detail": f"跨页一致水印本体（同色）：{body_detail}"}
    if body_v == "gray":
        return {"has_wm": True, "subtype": "gray", "detail": f"跨页一致水印本体（灰度）：{body_detail}"}
    if total_vote == 0:
        # 有水印但颜色不明确 -> 检查是否同色
        if _cross_page_same_color(acc, n_rendered, base_h, base_w):
            return {"has_wm": True, "subtype": "same", "detail": "有版心水印（同色，颜色判据失效走位置一致）"}
        return {"has_wm": True, "subtype": None, "detail": "有版心水印但颜色类不明（无法归类彩色/灰度/同色）"}
    half = max(1, int(total_vote * CROSS_PAGE_MIN))
    # 无彩色：仅剩灰度 vs 同色，取多数
    best = max([("gray", gray_n), ("same", same_n)], key=lambda kv: kv[1])
    if best[1] >= half:
        return {"has_wm": True, "subtype": best[0],
                "detail": f"{best[0]}水印（无彩色页，投票 {best[0]}={best[1]}/{total_vote}）"}
    # 投票分裂：取多数；若同色候选更可信
    if _cross_page_same_color(acc, n_rendered, base_h, base_w):
        return {"has_wm": True, "subtype": "same", "detail": "投票分裂但有跨页一致暗色（同色）"}
    return {"has_wm": True, "subtype": best[0],
            "detail": f"投票分裂取多数 {best[0]}（无彩色页，gray={gray_n},same={same_n}）"}


# ---------------- 顶层分类 ----------------
# —— CID 乱码/伪文本（乱码型）判定 ——
# 判据【单一源】= core/type_judge.py `_is_cid_garbled`（2026-10-24 用户定稿：把
# standard-pdf-cleaner 的判定迁移到分类技能统一维护，cleaner 引用本模块）：
# 取第 2 页起最多 4 页文本，汉字中属于高频常用字串 _CJK_COMMON 的比例 < 0.1 → CID 乱码/乱码型
# （复制不可用：get_text 汉字多为犐犆犛犌犅犜 这类生僻映射字，正常常用字占比极低）。
# 两个技能共用同一字表、同一阈值、同样取第 2 页起，保证同一文件在分类与去水印中判定一致。


def _open_doc_safe(path):
    """打开文档。返回 (doc 或 None, 是否需密码)。不重复渲染。"""
    return C.open_doc(path)


def classify_pdf_p1(path):
    """用途1（快速）：**先测打开（打不开=损坏）→ 再判定 乱码型/文本型/图像型**（v1.6.3 用户定稿统一顺序）。

    返回 (type_key, detail)。
      type_key: 'garbled' / 'text' / 'image'；损坏（打开失败/需密码/0页空文件）-> None（收容 04-其他）
    """
    doc, enc = _open_doc_safe(path)
    if enc or doc is None:
        return None, "打开失败或需密码（损坏打不开，未分类收容 04-其他）"
    try:
        n = doc.page_count
        if n == 0:
            # 先测打开：0 页空文件 = 损坏打不开，不归图像型（v1.6.3 用户定稿）
            return None, "0页空文件（损坏，无页面可读，未分类收容 04-其他）"
        kind = C.classify_text_image(doc)
        # 与去水印 cleaner 一致：文档含文本层但 CID 乱码/复制不可用 → 乱码型。
        # 注意放在 text/image 判定之前：混合型（如"前文本后扫描"）判 image 也须先查乱码。
        if C._is_cid_garbled(doc):
            return "garbled", "乱码型（CID乱码/复制不可用，伪文本）"
        if kind == "text":
            return "text", "文本型（文本层为主，无整页大图底）"
        return "image", "图像型/扫描件（需用途2细分水印）"
    finally:
        try:
            doc.close()
        except Exception:
            pass
        gc.collect()


def classify_pdf_p2(path):
    """用途2（细分）：**先测打开（打不开=损坏→05-损坏文件）→ 判类型 → 图像型再判版心水印子类**（v1.6.3 用户定稿统一顺序）。

    返回 (type_key, subtype, detail)。
      type_key: 'garbled' / 'text' / 'image'；损坏（打开失败/需密码/0页/无有效页可渲染）-> ('image','broken')
      subtype:  非图像型 None；图像型 'nowm' / 'color' / 'gray' / 'same' / 'broken'
    """
    doc, enc = _open_doc_safe(path)
    if enc or doc is None:
        # 打开失败/需密码 = 损坏打不开 → 05-损坏文件（v1.6.3：不再留在 03 根未分类）
        return "image", "broken", "打开失败或需密码（损坏，打不开）"
    try:
        n = doc.page_count
        if n == 0:
            # 0 页空文件：fitz 能打开但无任何页面（阅读器打不开）→ 05-损坏文件
            return "image", "broken", "0页空文件（损坏，无页面可读）"
        kind = C.classify_text_image(doc)
        # 与去水印 cleaner 一致：文档含文本层但 CID 乱码/复制不可用 → 乱码型。
        # 注意放在 text/image 判定之前：混合型（如"前文本后扫描"）判 image 也须先查乱码。
        if C._is_cid_garbled(doc):
            return "garbled", None, "乱码型（CID乱码/复制不可用，伪文本）"
        if kind == "text":
            return "text", None, "文本型（文本层为主，无整页大图底）"
        # 图像型：判定版心水印（渲染失败/无有效页 → broken，不进 nowm）
        verdict = image_wm_verdict(doc)
        if verdict.get("broken"):
            return "image", "broken", verdict["detail"]   # 无有效页可渲染（疑似损坏）→ 05-损坏文件
        if verdict["has_wm"]:
            sub = verdict["subtype"] or "color"  # 有版心水印但类不明时，默认归彩色并注明
            return "image", sub, verdict["detail"]
        # 无水印（nowm）：仅当 打开正常 + 渲染正常 + 版心内无非背景内容 + 无跨页一致暗色 才判
        return "image", "nowm", verdict["detail"]
    finally:
        try:
            doc.close()
        except Exception:
            pass
        gc.collect()


def classify_pdf(path, simple=False):
    """整合单一流程（v1.7.0 用户定稿）：**先测打开（打不开=损坏）→ 判类型 → 图像型自动细分水印**。

    simple=True（提示词说"简单分类"）：图像型不细分水印，直接归 03-图像型 根（省时）。

    返回 (type_key, subtype, detail)。
      type_key: 'garbled' / 'text' / 'image'；
      损坏（打开失败/需密码/0页/无有效页可渲染）-> ('image','broken')，归 03-图像型\05-损坏文件
      subtype:  非图像型 None；图像型 'color' / 'gray' / 'same' / 'nowm' / 'broken'（simple 时 None）
    """
    doc, enc = _open_doc_safe(path)
    if enc or doc is None:
        return "image", "broken", "打开失败或需密码（损坏，打不开，归 05-损坏文件）"
    try:
        n = doc.page_count
        if n == 0:
            return "image", "broken", "0页空文件（损坏，无页面可读，归 05-损坏文件）"
        kind = C.classify_text_image(doc)
        # v1.9.2 用户定稿：先判有无文本层——无文本层（图像页占比≥50%/平均<10字/页/有超大图）
        # 直接归图像型，不再跑乱码判定；有文本层（kind==text）才进一步分正常文本 vs CID乱码。
        if kind == "image":
            if simple:
                return "image", None, "图像型（简单分类，未细分水印）"
            verdict = image_wm_verdict(doc)
            if verdict.get("broken"):
                return "image", "broken", verdict["detail"]
            if verdict["has_wm"]:
                sub = verdict["subtype"] or "color"
                return "image", sub, verdict["detail"]
            return "image", "nowm", verdict["detail"]
        # kind == "text"：有文本层 → 再分正常文本 vs CID乱码
        if C._is_cid_garbled(doc):
            return "garbled", None, "乱码型（有文本层但CID乱码/复制不可用，伪文本）"
        return "text", None, "文本型（文本层正常，可复制）"
    finally:
        try:
            doc.close()
        except Exception:
            pass
        gc.collect()


def map_to_dir(type_key, subtype=None, simple=False):
    """类型 -> 目标子目录 key（整合 SUB_DIRS / 简单 SUB_DIRS_SIMPLE，v1.7.0）。

    simple=True：图像型归 03-图像型 根（"image"），不细分；
    默认：图像型细分水印。
    损坏（打开失败/需密码/0页/无有效页可渲染）-> 'image_broken'（03-图像型\05-损坏文件）；
    type_key is None（异常兜底）-> 'other'（04-其他）。
    """
    if type_key is None:
        return "other"
    if type_key == "garbled":
        return "garbled"
    if type_key == "text":
        return "text"
    # image
    if simple:
        return "image"  # 简单分类：图像型统一归 03-图像型 根
    if subtype == "color":
        return "image_wm_color"
    if subtype == "gray":
        return "image_wm_gray"
    if subtype == "same":
        return "image_wm_same"
    if subtype == "stripe":
        return "image_wm_stripe"   # 污带图像（扫描污带/整面底纹，v1.8 新增）→ 04-污带图像
    if subtype == "broken":
        return "image_broken"   # 0页空文件/无有效页可渲染（损坏，打不开）→ 06-损坏文件
    return "image_nowm"  # nowm 或 None
