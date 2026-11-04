# -*- coding: utf-8 -*-
"""把已规范命名的国标 PDF 按类型 + 年份复制到 Y:\\files_clean。

不移动、不删除源文件。目标已存在且大小相同则跳过。
启动时读取同目录的「续爬点.txt」和分类日志，已完成的文件直接跳过。
"""
import argparse
import csv
import os
import re
import shutil
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.classify import classify_pdf  # noqa: E402

SRC = r"Y:\磁盘阵列\标准文件下载\国标下载\国标-已规范名称"
DEST_ROOT = r"Y:\files_clean"
LOG_CSV = os.path.join(ROOT, "国标分类日志.csv")
STATUS_TXT = os.path.join(ROOT, "国标分类状态.txt")
CHECKPOINT = os.path.join(ROOT, "续爬点.txt")

YEAR_RE = re.compile(r"-(\d{4})(?:\s|$)")
SUBTYPE_DIR = {
    "color": "彩印型",
    "gray": "灰度型",
    "same": "同色型",
    "stripe": "污带型",
    "nowm": "无印型",
    "broken": "损坏型",
}
TYPE_LABEL = {"text": "文本型", "garbled": "乱码型", "image": "图像型"}
FIELDS = ["文件名", "年份", "判定类型", "图像子类", "目标路径", "状态", "耗时秒", "说明"]


def set_dest(path):
    global DEST_ROOT
    DEST_ROOT = path


def year_of(filename):
    base = os.path.splitext(filename)[0]
    m = YEAR_RE.search(base)
    if not m:
        return None
    return int(m.group(1))


def dest_dir_for(tkey, subtype, year):
    year_s = str(year)
    if tkey == "text":
        return os.path.join(DEST_ROOT, "文本型", "国标", year_s)
    if tkey == "garbled":
        return os.path.join(DEST_ROOT, "乱码型", "国标", year_s)
    sub = SUBTYPE_DIR.get(subtype or "color", "彩印型")
    return os.path.join(DEST_ROOT, "图像型", sub, "国标", year_s)


def is_done_status(status):
    return status == "已复制" or status.startswith("已存在跳过")


def load_done(path):
    done = set()
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if not is_done_status(row.get("状态") or ""):
                continue
            name = (row.get("文件名") or "").strip()
            if name.lower().endswith(".pdf"):
                done.add(name)
    return done


def load_checkpoint(path):
    done = set()
    if not os.path.isfile(path):
        return done
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            name = line.strip()
            if not name or name.startswith("#"):
                continue
            done.add(name)
    return done


def collect_done():
    return load_checkpoint(CHECKPOINT) | load_done(LOG_CSV)


def write_checkpoint(done):
    tmp = CHECKPOINT + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("# 国标分类续爬点\n")
        f.write(f"# 刷新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("# 每行一个已完成文件名。启动 classify_guobiao.py 会跳过这些文件。\n")
        f.write("# 只收录「已复制」和「已存在跳过」。判定失败的不在这里，下次会重试。\n")
        f.write(f"# 已完成 {len(done)}\n")
        for name in sorted(done):
            f.write(name + "\n")
    os.replace(tmp, CHECKPOINT)


def snapshot():
    done = collect_done()
    write_checkpoint(done)
    print(f"续爬点已写入 {CHECKPOINT} 共 {len(done)} 个", flush=True)
    return len(done)


def copy_file(src, dest_directory, name):
    os.makedirs(dest_directory, exist_ok=True)
    dest = os.path.join(dest_directory, name)
    src_size = os.path.getsize(src)
    if os.path.exists(dest):
        try:
            if os.path.getsize(dest) == src_size:
                return dest, "已存在跳过"
        except OSError:
            pass
        base, ext = os.path.splitext(name)
        i = 1
        while True:
            cand = os.path.join(dest_directory, f"{base}_{i}{ext}")
            if not os.path.exists(cand):
                dest = cand
                break
            try:
                if os.path.getsize(cand) == src_size:
                    return cand, "已存在跳过"
            except OSError:
                pass
            i += 1
    shutil.copy2(src, dest)
    return dest, "已复制"


def process_one(path):
    name = os.path.basename(path)
    year = year_of(name)
    t0 = time.perf_counter()
    if year is None:
        return {
            "文件名": name,
            "年份": "",
            "判定类型": "",
            "图像子类": "",
            "目标路径": "",
            "状态": "跳过:文件名无年代",
            "耗时秒": f"{time.perf_counter() - t0:.2f}",
            "说明": "",
        }
    try:
        tkey, subtype, detail = classify_pdf(path, simple=False)
    except Exception as e:
        return {
            "文件名": name,
            "年份": year,
            "判定类型": "",
            "图像子类": "",
            "目标路径": "",
            "状态": "判定失败",
            "耗时秒": f"{time.perf_counter() - t0:.2f}",
            "说明": f"{e}",
        }
    try:
        folder = dest_dir_for(tkey, subtype, year)
        dest, status = copy_file(path, folder, name)
    except Exception as e:
        return {
            "文件名": name,
            "年份": year,
            "判定类型": TYPE_LABEL.get(tkey, tkey or ""),
            "图像子类": subtype or "",
            "目标路径": "",
            "状态": "复制失败",
            "耗时秒": f"{time.perf_counter() - t0:.2f}",
            "说明": f"{detail} | {e}",
        }
    return {
        "文件名": name,
        "年份": year,
        "判定类型": TYPE_LABEL.get(tkey, tkey or ""),
        "图像子类": subtype or "",
        "目标路径": dest,
        "状态": status,
        "耗时秒": f"{time.perf_counter() - t0:.2f}",
        "说明": detail or "",
    }


def write_status(text):
    with open(STATUS_TXT, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 个未完成文件，0 表示全部")
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dest", default=DEST_ROOT)
    ap.add_argument("--snapshot", action="store_true", help="只根据日志刷新续爬点.txt，不开始分类")
    args = ap.parse_args()

    if args.snapshot:
        snapshot()
        return

    src = args.src
    set_dest(args.dest)
    workers = max(1, min(int(args.workers), 8))

    if not os.path.isdir(src):
        print(f"找不到源目录 {src}", flush=True)
        sys.exit(1)

    done = collect_done()
    names = [n for n in os.listdir(src) if n.lower().endswith(".pdf")]
    names.sort()
    pending = [os.path.join(src, n) for n in names if n not in done]
    if args.limit:
        pending = pending[: args.limit]

    new_log = not os.path.isfile(LOG_CSV)
    os.makedirs(ROOT, exist_ok=True)
    log_f = open(LOG_CSV, "a", encoding="utf-8-sig", newline="")
    ckpt_f = open(CHECKPOINT, "a", encoding="utf-8", newline="\n")
    writer = csv.DictWriter(log_f, fieldnames=FIELDS)
    if new_log:
        writer.writeheader()
        log_f.flush()

    lock = threading.Lock()
    total = len(pending)
    finished = 0
    copied = 0
    skipped = 0
    failed = 0
    t_start = time.perf_counter()
    print(
        f"续爬 已完成 {len(done)} 待处理 {total} workers {workers}",
        flush=True,
    )
    write_status(
        f"续爬开始 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"已完成 {len(done)}\n待处理 {total}\n"
    )

    batch = max(workers * 2, 8)

    def take(row):
        nonlocal finished, copied, skipped, failed
        with lock:
            writer.writerow(row)
            log_f.flush()
            name = row.get("文件名") or ""
            status = row["状态"]
            if name and is_done_status(status) and name not in done:
                done.add(name)
                ckpt_f.write(name + "\n")
                ckpt_f.flush()
            finished += 1
            if status == "已复制":
                copied += 1
            elif status.startswith("已存在跳过"):
                skipped += 1
            elif status in ("判定失败", "复制失败") or status.startswith("跳过"):
                failed += 1
            if finished % 20 == 0 or finished == total:
                elapsed = time.perf_counter() - t_start
                rate = finished / elapsed if elapsed else 0
                eta = (total - finished) / rate if rate else 0
                line = (
                    f"完成 {finished}/{total} 复制 {copied} 跳过 {skipped} 失败 {failed} "
                    f"速度 {rate:.2f}/秒 已用 {elapsed/60:.1f}分 剩余约 {eta/60:.0f}分"
                )
                print(line, flush=True)
                write_status(
                    line
                    + f"\n更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    + f"最近文件 {row['文件名']}\n最近去向 {row['目标路径']}\n"
                )

    try:
        if workers == 1:
            for path in pending:
                take(process_one(path))
        else:
            pos = 0
            with ThreadPoolExecutor(max_workers=workers) as ex:
                while pos < total:
                    futs = []
                    while pos < total and len(futs) < batch:
                        futs.append(ex.submit(process_one, pending[pos]))
                        pos += 1
                    for fut in as_completed(futs):
                        try:
                            take(fut.result())
                        except Exception:
                            take({
                                "文件名": "",
                                "年份": "",
                                "判定类型": "",
                                "图像子类": "",
                                "目标路径": "",
                                "状态": "判定失败",
                                "耗时秒": "",
                                "说明": traceback.format_exc(limit=2),
                            })
    finally:
        log_f.close()
        ckpt_f.close()
        if finished:
            try:
                write_checkpoint(collect_done())
            except OSError:
                pass
    elapsed = time.perf_counter() - t_start
    end = (
        f"结束 完成 {finished}/{total} 复制 {copied} 跳过 {skipped} 失败 {failed} "
        f"用时 {elapsed/60:.1f}分"
    )
    print(end, flush=True)
    write_status(end + f"\n更新时间 {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


if __name__ == "__main__":
    main()
