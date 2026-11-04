# 国标分类（可续爬）

把 `Y:\磁盘阵列\标准文件下载\国标下载\国标-已规范名称` 里的国标 PDF，按类型和年份复制到 `Y:\files_clean`。只复制，不移动、不删除源文件。

判定逻辑在 `core/`，与 standard-pdf-classifier 1.9.2 相同。

## 同事接着跑

同一台能访问 `Y:` 盘的电脑：

```
pip install -r requirements.txt
python classify_guobiao.py
```

启动后会读取同目录的 `续爬点.txt` 和 `国标分类日志.csv`，已经复制过的文件直接跳过，从剩下的接着处理。目标位置如果已经有同名且大小相同的文件，也会跳过，不会覆盖。

盘符不同时：

```
python classify_guobiao.py --src "Y:\磁盘阵列\标准文件下载\国标下载\国标-已规范名称" --dest "Y:\files_clean"
```

## 推送到 GitHub 之前

分类如果还在跑，不要再开第二个分类进程。推送前先把最新进度收成续爬点：

```
python classify_guobiao.py --snapshot
```

这条命令只读日志，把已完成文件名写入 `续爬点.txt`，不会开始分类。然后把 `续爬点.txt` 和 `国标分类日志.csv` 一起提交。同事拉下来再执行 `python classify_guobiao.py` 即可续爬。

`续爬点.txt` 每行一个已完成文件名，只收录「已复制」和「已存在跳过」。判定失败的不在里面，下次启动会重试。
