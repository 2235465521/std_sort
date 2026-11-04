# -*- coding: utf-8 -*-
"""standard-pdf-classifier / core/type_judge.py —— 类型判定【单一源】模块（v1.4.0）

文本型 / 乱码型 / 图像型的判定（乱码判定 `_is_cid_garbled`、文本/图像判定 `is_text_pdf`
及其全部判据常量）**只在本模块定义**，供两个技能引用，避免重复维护导致漂移：
  - standard-pdf-classifier：core/common.py、core/classify.py 直接 `from core.type_judge import ...`；
  - standard-pdf-cleaner：core/common.py 通过 importlib 按文件路径加载本模块后转发使用。

判据来源（2026-10-24 用户定稿）：**以 standard-pdf-cleaner 的水印清除技能现有判定为准**——
cleaner 判定经大量真实样本调优（v1.21.26/27/43：活氧多效清洁剂、有机硅密封胶、活性炭、
香皂、井盖等），实际验证更准，故把该判定整体迁移到本模块，classifier 与 cleaner 共用同一份。

设计约束（对齐用户强规则）：
  1. 本模块【自包含】——只依赖 stdlib（re/random）与 fitz（pymupdf），
     不 import 任何包内模块（保证跨技能 importlib 加载时无包名冲突）；
  2. 判定以文本级/版面几何统计为准（可复核、可复现），不依赖视觉模型文字描述；
  3. 命名定稿：乱码型 = 复制为乱码（CID 伪文本）；需密码/打开失败【不归乱码型】（由调用方判未分类）。
"""
import re
import random

import fitz

# ---------------- 判据常量（以 standard-pdf-cleaner 定稿值为准，2026-10-24） ----------------
TEXT_MIN_TOTAL_CHARS = 100     # 文本型 PDF 判定阈值：全文可提取字符数下限（历史保留，语义文档用）
TEXT_MIN_PER_PAGE = 10         # 平均每页字符数下限：抽样页平均文本量低于此值 → 图像型
LARGE_IMAGE_BYTES = 5 * 1024 * 1024  # 整页级超大图阈值（字节）
SAMPLE_MAX_PAGES = 10          # 抽样页上限（第2页起、排除末页；固定种子可复现）
_SAMPLE_SEED = 20261011        # 抽样固定种子（cleaner 定稿值）
TEXT_PER_PAGE_LOW = 200        # 低文本密度阈值：每页平均文本低于此值视为"碎片文本层"（正文不在文本层）
IMAGE_PRESENT_THRESH = 0.2     # 图像"存在"阈值：文本密度低时，抽样页存在图像覆盖超过此值 → 佐证图像型
IMAGE_COVER_MIXED_THR = 0.5    # 混合型判据：图像覆盖率超过此值 且 该页文本碎片 → 该页为"图像页"
IMAGE_PAGE_RATIO_THR = 0.5     # 整体页面情况阈值（v1.6.5 用户定稿）：图像页在抽样页中占比 ≥ 此值 → 图像型
                               # （个别大插图页不判图像：整体多数页是图像才算图像型）


# ---------------- 乱码判定（乱码型 = 复制为乱码 / CID 伪文本） ----------------
# 高频常用字串（复制自 standard-pdf-cleaner cleaner.py，两技能共用同一字表）：
# 汉字中属于该串的比例 < 0.1 → CID 生僻映射字乱码（犐犆犛犌犅犜…）
_CJK_COMMON = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"


def _is_cid_garbled(doc):
    """文本层是否 CID 乱码/伪文本（复制不可用）。判据与 standard-pdf-cleaner cleaner.py `_is_cid_garbled` **完全一致**：

      1) 跳过封面页（第1页常为正常 CID 大字标题，拉高汉字占比掩盖正文乱码——GB 12021.9 实测），
         从第 2 页起取最多 4 页正文页；单页文档无正文页 → 直接不算乱码（v1.5.0 防御修复，避免越界）；
      2) CJK 判据优先：正文文本足够（≥50 字符）但汉字占比 <10% → Type1 子集无 ToUnicode、正文汉字输出
         ASCII 符号（"范"→'.'，不在 FFFD/全角/私有区）→ 真乱码（旧判据因汉字为空 return False 漏判）；
      3) 汉字中常用字占比 <0.1 → CID 生僻映射字（犐犆犛犌犅犜…）。

    返回 True=乱码/伪文本（复制不可用）。需密码/打开失败【不归乱码型】（调用方判未分类，留输入根）。
    """
    try:
        n = doc.page_count
        start = 1
        count = min(4, n - 1)          # 单页文档无正文页（count=0）→ 直接不算乱码，避免 doc[1] 越界
        if count <= 0:
            return False
        idxs = list(range(start, start + count))
        t = "".join(doc[i].get_text() for i in idxs)
    except Exception:
        return False
    total = len(re.sub(r"\s", "", t))
    han = re.findall(r'[\u4e00-\u9fff]', t)
    if total >= 50 and len(han) / total < 0.10:
        return True
    if not han:
        return False
    common = sum(1 for c in han if c in _CJK_COMMON)
    return common / len(han) < 0.1


# ---------------- 文本型 / 图像型判定 ----------------
def page_image_coverage(page):
    """页面被图像 bbox 覆盖的面积占比（0~1）。用于识别"正文是整页扫描图"的图像型 PDF。
    实现与 standard-pdf-cleaner common.py `page_image_coverage` 一致（get_image_info bbox）。
    """
    try:
        info = page.get_image_info()
    except Exception:
        return 0.0
    if not info:
        return 0.0
    pw, ph = page.rect.width, page.rect.height
    area = 0.0
    for im in info:
        r = im.get("bbox")
        if r:
            w = max(0, r[2] - r[0]); h = max(0, r[3] - r[1])
            area += w * h
    return min(1.0, area / max(1.0, pw * ph))


def has_large_image(doc, threshold=LARGE_IMAGE_BYTES):
    """是否含整页级超大图（单图原始字节 > threshold）。识别"整页扫描图 + 平台叠加/OCR文字层"：
    这类文件可能文本量达标，但页面主体是超大扫描图。实现与 cleaner common.py `has_large_image`
    一致（遍历全部页、xref_stream_length 判字节）。返回 bool。
    """
    try:
        for page in doc:
            for im in page.get_images(full=True):
                xref = im[0]
                try:
                    ln = doc.xref_stream_length(xref)
                except Exception:
                    ln = 0
                if ln > threshold:
                    return True
    except Exception:
        pass
    return False


def _sample_pages(doc):
    """抽样页：第2页..倒数第2页；页数<=2 退化为全部页。固定种子可复现（cleaner 定稿：SAMPLE_MAX_PAGES=10）。"""
    n = doc.page_count
    if n == 0:
        return []
    cand = list(range(1, n - 1)) if n > 2 else list(range(n))
    if not cand:
        return []
    if len(cand) > SAMPLE_MAX_PAGES:
        rng = random.Random(_SAMPLE_SEED)
        cand = rng.sample(cand, SAMPLE_MAX_PAGES)
    return cand


def is_text_pdf(doc):
    """判断 PDF 是否以文本层为主（可分类为文本型/可安全执行清理）。与 cleaner common.py `is_text_pdf` **完全一致**：

      图像型（扫描件/纯图）无可提取文本，返回 False。增强判据（cleaner v1.21.26/27/43 调优 + v1.6.5 整体页面判据）：
        * 含整页级超大图（>5MB）→ 图像型；
        * 抽样页平均文本量过低（<10 字/页）→ 图像型；
        * 【v1.6.5 用户定稿：考虑整体页面情况】图像页（覆盖率>0.5 且 该页文本<200）在抽样页中占比 ≥ 0.5
          （多数页是图像）→ 图像型；**个别大插图页不再触发整文档判图像型**
          （实测 GB_T_25517.1/14926.30/1979 等"文本为主+技术插图/产品照片"曾被旧"任一页命中"判据误判）。
        * 抽样页：第2页..倒数第2页，>10 页时固定种子随机抽 ≤10 页（避免封面/末页干扰、可复现）。

    返回: bool。
    """
    n = doc.page_count
    if n == 0:
        return False
    cand = _sample_pages(doc)
    if not cand:
        return False
    texts = []
    covs = []
    for i in cand:
        texts.append(len(doc[i].get_text().strip()))
        covs.append(page_image_coverage(doc[i]))
    avg_text = sum(texts) / len(texts)
    # v1.6.5：整体页面情况——图像页占比 ≥ 0.5（多数页是图像）才判图像型；
    # 取代旧判据（任一抽样页 cov>0.5 且 text<200 → 图像型；avg_text<200 且 max_cov>0.2 → 图像型）
    img_pages = sum(1 for t, c in zip(texts, covs)
                    if c > IMAGE_COVER_MIXED_THR and t < TEXT_PER_PAGE_LOW)
    if img_pages / len(texts) >= IMAGE_PAGE_RATIO_THR:
        return False
    # 抽样页平均文本量过低 → 图像型
    if avg_text < TEXT_MIN_PER_PAGE:
        return False
    # 文本达标但含整页级超大图 → 图像型
    if has_large_image(doc):
        return False
    return True
