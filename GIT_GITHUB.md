# Git 与 GitHub 发布新版本

本文针对当前工程给出可直接执行的 PowerShell 流程。当前分支是 `main`，远端名称是 `origin`，远端地址为：

```text
git@github.com:ChiaKi707/classify.git
```

Git 只上传源码、配置、测试和说明文档。原始数据、准备数据、模型权重与训练结果由 `.gitignore` 排除，不应提交到公开仓库。

## 1. 发布前检查

在工程根目录执行：

```powershell
git status --short
git diff --check
python -m unittest discover -s tests -p "test_*.py"
python check.py
git check-ignore -v data data_prepared data_plain output pt
```

先确认终端已激活项目环境，通常提示符会显示 `(pytorch)`。如果 `python` 报告没有 `torch`，说明当前调用的是系统解释器；请先激活 Conda 环境再重试：

```powershell
conda activate pytorch
python --version
python -c "import torch; print(torch.__version__)"
```

检查重点：

- `git diff --check` 没有空白或补丁格式错误；
- 单元测试和 `check.py` 通过；`check.py` 只使用临时合成数据，不跑正式训练；
- 最后一条命令能显示这些本地数据和结果目录对应的 `.gitignore` 规则；
- `git status --short` 中不出现真实 CSV、模型权重、未知样本、密钥或账号文件。

本次整理会删除历史入口 `run_all.py` 和过期报告 `PROJECT_REVIEW_2026-10-17.md`，由 `run_experiments.py`、当前五份主题文档和本文件替代。后面的 `git add -A` 会把这两个删除也加入提交，因此必须在提交前核对暂存区。

## 2. 暂存并检查本次版本

```powershell
git add -A
git status --short
git diff --cached --stat
git diff --cached
```

如果误暂存了某个文件，可只撤销暂存而保留本地内容：

```powershell
git restore --staged 路径
```

不要把数据、权重或密钥强制加入 Git。GitHub 官方也明确建议在推送前移除密码、API 密钥等敏感信息，参见 [Adding a file to a repository](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)。

## 3. 创建本地提交

确认暂存内容正确后：

```powershell
git commit -m "refactor: unify data protocol and model comparisons"
```

此时版本已经保存在本地 Git，但还没有上传到 GitHub。

## 4. 同步并推送到当前 GitHub 仓库

先确认远端和 SSH 身份：

```powershell
git remote -v
ssh -T git@github.com
```

SSH 首次连接可能要求确认主机指纹；认证成功后 GitHub 通常会提示认证成功但不提供 shell。官方说明见 [Testing your SSH connection](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/testing-your-ssh-connection)。

然后同步远端并推送：

```powershell
git pull --rebase origin main
git push origin main
```

`pull --rebase` 用于在推送前接入 GitHub 上可能新增的提交。若出现冲突，不要删除文件或强制推送；先执行 `git status`，逐个解决冲突后再继续：

```powershell
git add 冲突文件
git rebase --continue
git push origin main
```

GitHub 官方的本地代码发布流程同样使用“添加远端 → 验证远端 → 推送主分支”，参见 [Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。本工程已经配置好 `origin`，不需要再次执行 `git remote add origin`。

## 5. 可选：标记一个正式版本

完整实验协议稳定后，可以创建带说明的标签：

```powershell
git tag -a v2.0.0 -m "Unified split, augmentation, ablation and model comparison protocol"
git push origin v2.0.0
```

标签适合标记论文实验所使用的代码快照。每份训练结果还保存数据哈希、代码指纹和有效配置，用于确认结果是否确实来自该版本。

## 6. 新仓库时才需要的命令

只有未来把项目发布到另一个空仓库时，才需要替换远端：

```powershell
git remote set-url origin git@github.com:<用户名>/<新仓库>.git
git push -u origin main
```

不要在当前仓库正常更新时重复执行 `git init`，也不要重复添加同名 `origin`。
