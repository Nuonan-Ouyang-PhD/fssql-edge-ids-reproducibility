# 2026-10-22 仓库更新说明

本次更新依据已完成的 Pi4B R6，更新分析、测试、正文、补充材料、回复信和复现说明。不修改冻结的运行代码、配置、模型、Q 表、执行清单或原始测量，不启动硬件，不重新访问官方测试集。

## 更新内容

- `scripts/analyze_pi4b8_r6.py` 修正 FPR 被前缀过滤误排除的问题，并复核 60 个接受 run 的评分、奖励、Q 选择、功耗积分和状态支持。
- 当前分析在 `results/pi4b8_r6_revision/`。
- `paper/` 包含当前投稿稿件：主文 8 页，回复信 14 页，补充材料 4 页，投稿信 1 页。
- guard 浮点复算采用很小的数值容差，并严格检查 coverage/admission 决策一致；冻结文件本身仍逐字节验 SHA-256。
- 本地 118 项测试通过。所有 60 个 R6 功率积分和 18 个 FPR 配对对比已检查。

## 应用与复核

先使用更新包的 `apply_update.py` 进行 dry-run，确认目标仓库确为本篇 TSUSC Major Revision，且待替换文件仍匹配提供包的基准哈希。不要应用到另一个 `fssql-experiment` 项目。若当前仓库已有额外修改，逐项比较并人工合并，不要强行覆盖。

应用后执行：

```bash
python scripts/validate_public_release.py --with-evidence
python -m unittest discover -s tests -v
python scripts/analyze_pi4b8_r6.py --output /tmp/tsusc-r6-independent-check
```

公共 checkout 缺少原始证据时，只执行不带 `--with-evidence` 的发布检查；需要原始文件的测试必须使用原有完整归档，不能凭空补齐。

检查 `git diff` 后，以一个正常的新 commit 更新 README、当前分析、文稿、脚本与测试。不得 force-push、重写旧提交或删除失败实验来隐藏更正。GitHub 更新不是重新验收历史无效结果。

不要把原始 UNSW-NB15 数据、私有输入包、凭据、设备访问配置或未获发布许可的文件直接 `git add -A`。只暂存经检查的更新文件。不要新增不符合原数据许可的仓库许可证。

## 投稿文件

投稿系统使用 `paper/TSUSC-2026-07-0200-main.pdf`、`-supp.pdf`、`-response.pdf` 和 `-cover-letter.pdf`。独立投稿 ZIP 中 `main.tex` 位于根目录；编译以其为主文件。所有表图均来自当前 R6 分析。

提交前请共同作者确认作者信息、投稿系统元数据、修订内容及补充材料。
