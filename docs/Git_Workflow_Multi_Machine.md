# 多机开发 Git 工作流说明

本文档用于说明：当你在另一台机器上继续开发 `dexbotic` 时，如何安全地拉取、开发、提交并推送到你自己账号的 `dev` 分支，同时尽量避免本次会话里遇到的几类常见问题：

- VS Code Git 凭据代理失效
- rebase 时未跟踪文件阻塞
- 目录权限异常导致 rebase / checkout 失败
- 临时调试产物误入提交
- 多台机器开发时远端 `dev` 分支快进失败

适用前提：

- 个人开发分支：`dev`
- 可能存在两种常见 remote 布局

---

## 1. 两种常见 remote 结构

多机开发时，先分清你当前机器属于哪一种 remote 结构。

### 场景 A：本地是从官方仓库 clone 下来的

这种情况下通常是：

- `origin` -> 官方仓库
- `myfork` -> 你自己的 fork

例如：

```text
origin  https://github.com/Dexmal/dexbotic.git
myfork  https://github.com/<YOUR_GITHUB_USERNAME>/dexbotic.git
```

### 场景 B：本地是直接从你自己的 fork clone 下来的

这是你当前另一台机器更符合的场景。

这种情况下通常是：

- `origin` -> 你自己的 fork
- `upstream` -> 官方仓库

例如：

```text
origin   https://github.com/<YOUR_GITHUB_USERNAME>/dexbotic.git
upstream https://github.com/Dexmal/dexbotic.git
```

**本文后续默认优先按场景 B 来写**，因为它更适合你当前的多机开发方式。

---

## 2. 推荐的 remote 结构

### 2.1 场景 B：默认推荐结构

如果另一台机器是从你的 fork 直接 clone 下来的，推荐保持：

```bash
git remote -v
```

应类似于：

```text
origin    https://github.com/<YOUR_GITHUB_USERNAME>/dexbotic.git (fetch)
origin    https://github.com/<YOUR_GITHUB_USERNAME>/dexbotic.git (push)
upstream  https://github.com/Dexmal/dexbotic.git (fetch)
upstream  https://github.com/Dexmal/dexbotic.git (push)
```

如果还没有 `upstream`，补上：

```bash
git remote add upstream https://github.com/Dexmal/dexbotic.git
```

如果你使用 SSH：

```bash
git remote add upstream git@github.com:Dexmal/dexbotic.git
```

### 2.2 场景 A：官方仓库 + 个人 fork

如果当前机器是从官方仓库 clone 下来的，则推荐：

```text
origin  https://github.com/Dexmal/dexbotic.git
myfork  https://github.com/<YOUR_GITHUB_USERNAME>/dexbotic.git
```

如果缺少 `myfork`：

```bash
git remote add myfork https://github.com/<YOUR_GITHUB_USERNAME>/dexbotic.git
```

---

## 3. 新机器首次配置

### 2.1 配置 Git 身份

避免提交作者名变成系统自动生成值：

```bash
git config --global user.name "你的 GitHub 用户名或常用名字"
git config --global user.email "你的 GitHub 邮箱"
```

### 2.2 配置认证

如果你使用 GitHub token，推荐：

```bash
git config --global credential.helper store
```

以后推送时，首次输入一次 token 即可。

如果 VS Code 的 Git 凭据代理在这台机器上不稳定，推送时优先使用：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push ...
```

这样会直接走终端输入，不依赖 VS Code 的 socket。

---

## 4. 开始开发前的标准动作

无论在哪台机器，开始开发前都建议按下面顺序执行。

### 4.1 获取远端最新状态

```bash
git fetch origin
```

如果你使用的是场景 B，再补：

```bash
git fetch upstream
```

如果你使用的是场景 A，再补：

```bash
git fetch myfork
```

### 4.2 切到你的开发分支

如果本地还没有 `dev`：

场景 B：

```bash
git checkout -b dev origin/dev
```

场景 A：

```bash
git checkout -b dev myfork/dev
```

如果本地已有：

```bash
git checkout dev
```

### 4.3 先同步远端 `dev`

推荐用 rebase：

场景 B：

```bash
git rebase origin/dev
```

场景 A：

```bash
git rebase myfork/dev
```

或者如果你更喜欢 merge：

```bash
git merge myfork/dev
```

推荐优先 `rebase`，历史更干净。

---

## 5. 如果你还想同步官方仓库

这一步主要用于场景 B，即：

- `origin` 是你的 fork
- `upstream` 是官方仓库

如果你想把官方主仓库 `main` 的新变化带到当前机器：

```bash
git fetch upstream
git checkout main
git rebase upstream/main
```

然后再把这些变化带回你的开发分支：

```bash
git checkout dev
git rebase main
```

如果你的 `main` 也希望和你自己 fork 的 `origin/main` 对齐，可以后续再推：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push origin main
```

---

## 6. 开发期间的目录约定

当前仓库已经整理出以下结构：

- `data_tools/`：数据转换与数据准备
- `openloop/`：open-loop 评估、诊断工具与评估产物
- `playground/post_data_01_dm0_deltafix.py`：当前推荐的训练入口
- `docs/`：开发报告与流程文档

开发时建议遵守：

### 4.1 代码放哪里

- 数据转换脚本放 `data_tools/`
- open-loop 评估与调试脚本放 `openloop/tools/`
- 训练 benchmark / 实验入口放 `playground/`
- 文档放 `docs/`

### 4.2 不要把评估产物和训练产物加进 Git

这些目录已经在 `.gitignore` 中排除，不要手工 `git add -f`：

- `openloop/artifacts/`
- `wandb/`
- `user_checkpoints/`
- `data/`
- `checkpoints/`

---

## 7. 完成开发后的标准提交流程

### 5.1 先看状态

```bash
git status
```

### 5.2 检查是否有不该提交的内容

重点留意：

- `__pycache__/`
- `*.pyc`
- 日志文件
- `openloop/artifacts/`
- `wandb/`
- 本地临时目录

如果 `git status` 里出现了这类文件，先确认是否应该进入提交。

### 5.3 暂存你真正要提交的改动

推荐显式写路径，而不是直接 `git add .`：

```bash
git add \
  .gitignore \
  data_tools \
  openloop \
  playground \
  docs \
  dexbotic/exp/base_exp.py \
  dexbotic/exp/dm0_exp.py
```

再看一眼：

```bash
git status
```

### 5.4 提交

```bash
git commit -m "feat: 描述这次改动"
```

---

## 8. 推送到你账号的 `dev` 分支

### 8.1 场景 B：`origin` 就是你的 fork

直接推送：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev
```

### 8.2 场景 A：`myfork` 是你的 fork

直接推送：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u myfork dev
```

如果远端 `dev` 比你本地更新，Git 会拒绝推送。这时不要第一时间 `force push`，先同步。

场景 B：

```bash
git fetch origin
git rebase origin/dev
```

场景 A：

```bash
git fetch myfork
git rebase myfork/dev
```

如果 rebase 成功，再推。

场景 B：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev
```

场景 A：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u myfork dev
```

如果你已经提交过、又做了 rebase，导致本地 commit hash 改变，则推送要用。

场景 B：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev --force-with-lease
```

场景 A：

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u myfork dev --force-with-lease
```

注意：优先使用 `--force-with-lease`，不要直接用 `--force`。

---

## 9. 多机开发最推荐的节奏

推荐每次上机都按这个节奏：

### 9.1 场景 B：另一台机器直接 clone 你的 fork

#### 开始前

```bash
git fetch origin
git fetch upstream
git checkout dev
git rebase origin/dev
```

#### 如果要同步官方主仓库

```bash
git checkout main
git rebase upstream/main
git checkout dev
git rebase main
```

#### 开发完成后

```bash
git status
git add ...
git commit -m "..."
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev
```

#### 如果发生 rebase 改历史

```bash
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev --force-with-lease
```

### 9.2 场景 A：官方仓库 + 个人 fork

```bash
git fetch myfork
git checkout dev
git rebase myfork/dev
```

开发完成后：

```bash
git status
git add ...
git commit -m "..."
GIT_ASKPASS= SSH_ASKPASS= git push -u myfork dev
```

这样两台机器始终围绕你的个人 `dev` 分支串行推进，不容易分叉失控。

---

## 10. 遇到 rebase 阻塞时怎么处理

本次会话里踩过的坑主要有 3 种。

### 8.1 未跟踪文件阻塞 rebase

报错典型形式：

```text
The following untracked working tree files would be overwritten by merge
```

处理原则：

- 不要先删，先备份
- 先判断这个文件是否真的应该进入当前提交

推荐做法：

```bash
mkdir -p /tmp/rebase_backup
cp <file> /tmp/rebase_backup/
rm <file>
git rebase --continue
```

### 8.2 目录或文件权限不对，导致 Git 无法写入

报错典型形式：

```text
unable to create file ... Permission denied
```

处理：

```bash
sudo chown -R $(whoami):$(whoami) <目录>
```

例如：

```bash
sudo chown -R $(whoami):$(whoami) playground/benchmarks/custom
```

### 8.3 rebase todo 里重复 replay 同一个 commit

如果看到：

- 同一个 `pick <commit>` 重复出现多次
- `git status` 显示 interactive rebase in progress

处理：

```bash
git rebase --edit-todo
```

删掉重复的 `pick` 行，然后：

```bash
git rebase --continue
```

---

## 11. 如何判断文件是不是“rebase 丢了”

建议按下面顺序查：

### 9.1 看原始提交是否包含该文件

```bash
git show --name-only <old_commit>
```

### 9.2 看 rebase 后新提交是否还包含该文件

```bash
git show --name-only HEAD
```

### 9.3 看当前分支树里是否存在该文件

```bash
git ls-tree -r --name-only HEAD | grep '<path>'
```

如果原提交里有、新提交里没了，基本就是 rebase 过程中掉了。

---

## 12. 建议保留的本地调试习惯

### 10.1 推送前先确认工作区干净

```bash
git status
```

### 10.2 定期清理本地缓存

```bash
find . -type d -name '__pycache__' -prune -exec rm -rf {} +
find . -type f -name '*.pyc' -delete
```

### 10.3 日志文件不进 Git

训练日志、调试日志、评估产物一律视为本地文件，不要加入提交。

---

## 13. 当前项目的推荐入口

### 数据转换

```bash
python data_tools/convert_post_data_01_to_dexdata.py ...
python data_tools/convert_post_data_01_to_dexdata_stateful.py ...
```

### 训练

```bash
torchrun --nproc_per_node=4 playground/post_data_01_dm0_deltafix.py
```

### open-loop 评估

```bash
python openloop/eval_openloop.py ...
```

### 开发说明

- [playground/post_data_01_dm0_deltafix_README.md](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix_README.md:1)
- [docs/DM0_Openloop_Debug_Report_post_data_01.md](/home/guoyaokun/dexbotic/docs/DM0_Openloop_Debug_Report_post_data_01.md:1)

---

## 14. 一套最短可复制流程

在另一台机器上，从开始到推送，最短流程建议如下：

### 场景 B：另一台机器的 `origin` 就是你的 fork

```bash
git fetch origin
git fetch upstream
git checkout dev || git checkout -b dev origin/dev
git rebase origin/dev
```

开发完成后：

```bash
git status
git add <你真正改动的路径>
git commit -m "feat: your change"
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev
```

如果提示远端领先：

```bash
git fetch origin
git rebase origin/dev
GIT_ASKPASS= SSH_ASKPASS= git push -u origin dev --force-with-lease
```

### 场景 A：当前机器的 `myfork` 是你的 fork

```bash
git fetch origin
git fetch myfork
git checkout dev || git checkout -b dev myfork/dev
git rebase myfork/dev
```

开发完成后：

```bash
git status
git add <你真正改动的路径>
git commit -m "feat: your change"
GIT_ASKPASS= SSH_ASKPASS= git push -u myfork dev
```

如果提示远端领先：

```bash
git fetch myfork
git rebase myfork/dev
GIT_ASKPASS= SSH_ASKPASS= git push -u myfork dev --force-with-lease
```

---

## 15. 最终建议

多机开发时最重要的不是“命令记住多少”，而是保持两个习惯：

1. 开发前先 `fetch + rebase`
2. 提交时只 `add` 明确需要进入 Git 的路径

只要坚持这两点，就能显著减少：

- 远端快进失败
- rebase 掉文件
- 调试产物误提交
- 不同机器状态漂移
