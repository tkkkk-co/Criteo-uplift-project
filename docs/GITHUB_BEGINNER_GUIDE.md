# 从本机项目到GitHub：第一次发布

## 四个概念

| 概念 | 作用 |
|---|---|
| 项目目录 | 当前磁盘上的代码、文档和结果 |
| Git | 本机记录文件版本和提交历史的工具 |
| GitHub | 托管Git仓库的网站 |
| Python环境 | 解释器及依赖，负责运行代码，与Git仓库是不同对象 |

Git不训练模型，GitHub链接也不会自动保存所有本地文件。`commit`在本机记录一个版本；`push`将已提交的版本上传到远端。README介绍项目，.ipynb展示分析，.py保存复用逻辑。pyproject定义安装，tests检查程序正确性。

项目根目录为包含README、pyproject、criteo_uplift等内容的最外层目录。`sys.path` 是Python的模块搜索列表；将根目录放在列表前部，便于 `import criteo_uplift`。这不改变当前工作目录。editable安装后，包可以由环境找到，相关解释见复现文档。

## 推荐先用GitHub Desktop

如果当前不习惯终端，可以使用官方GitHub Desktop，登录GitHub账户，将本机项目加入/创建为本地仓库，查看Changes，写首个commit摘要，再Publish repository。首次公开发布时确认Public设置；之后修改文件→查看差异→Commit→Push origin。

如果目录不是Git仓库，Desktop可能提示创建仓库。指定已存在项目作为本地路径，不要意外再套一层同名文件夹。发布前确认Changes中没有data、artifacts、.runtime、.env。这些排除规则已经写在.gitignore；公开聚合结果在reports/，应该提交。

官方操作参考：[从本机添加仓库](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-a-repository-from-your-local-computer-to-github-desktop)、[发布本地仓库](https://docs.github.com/en/desktop/adding-and-cloning-repositories/adding-an-existing-project-to-github-using-github-desktop)。

## 终端方式：从头到尾

以下用户名、邮箱与URL都是需要自己替换的示例，不是已配置的账户。

### 1. 进入目录，初始化本机版本库

```bash
cd ~/Desktop/Criteo-uplift-project
git --version
git init -b main
```

`cd`修改终端的当前目录。`git init`创建.git元数据，开始记录版本；重复初始化现有仓库不是重新删除代码。main是分支名。

### 2. 配置提交署名，仅对本仓库

```bash
git config user.name '你的GitHub显示名'
git config user.email '你的已验证邮箱或GitHub提供的noreply邮箱'
```

署名不等于登录凭据。不要把示例原封不动写进去，也不要将密码/token放进项目或remote URL。

### 3. 检查并提交

```bash
python scripts/check_publication.py
git status --short
git add .
git diff --cached --stat
git ls-files data artifacts .runtime
git commit -m 'Add reproducible uplift policy study'
```

`git add .`把未被忽略的改动加入暂存区；暂存区是下一次commit的候选内容。`git diff --cached --stat`查看将提交哪些文件；`git ls-files data artifacts .runtime`应无输出。`commit`记录本机版本，但仍未上传。

.gitignore不会删除本机文件，也不会自动撤销已经被Git跟踪的文件。首次提交时正确排除即可避免把3GB数据加进去。先查看Changes/暂存结果再提交。

### 4. 在GitHub网页创建空仓库

登录后点击New repository。建议名称 `criteo-uplift-policy-evaluation`，描述例如：

```text
Incremental conversion targeting with causal learners, budgeted policy evaluation and paired uncertainty.
```

用于简历展示可选择Public；若要先检查，可先Private。因为本机已有README、许可证与.gitignore，网页创建时先不要添加这些初始化文件，以免产生两套首个历史。[GitHub官方本机导入指南](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)

### 5. 把本机仓库连接到远端并上传

复制新仓库的HTTPS地址，替换示例：

```bash
git remote add origin https://github.com/YOUR_USERNAME/criteo-uplift-policy-evaluation.git
git remote -v
git push -u origin main
```

origin是为远端地址取的名字；main是要推送的分支；`-u`记录关联，以后通常直接 `git push`。如果origin已经配置，先查看它，不重复add；确认要更换后使用 `git remote set-url origin 新URL`。

首次连接需要认证。HTTPS可以通过GitHub Desktop/credential helper或适当token；GitHub账户密码不是Git push的认证方式。SSH需要密钥配置，对新手可先使用Desktop。[官方远端与认证说明](https://docs.github.com/en/get-started/git-basics/about-remote-repositories)

### 6. 检查远端结果与后续更新

网页确认README、Notebook、公开表格/图、理论笔记出现；Actions检查测试、演示与Notebook执行。CI若失败，以日志为准，不据本机测试自动声称远端通过。HTML下载本地查看。

以后修改代码后：

```bash
git status
git add .
git diff --cached --stat
git commit -m 'Explain or implement the actual change'
git push
```

在另一台电脑获取：`git clone 仓库HTTPS地址`，再按README创建Python环境。浏览器下载ZIP也能获得源码，但没有本机Git历史，需要另行建立版本库。

## 本次准备的发布材料

整个工作区已经按上述结构整理；`scripts/build_release.py`可生成只含代码/文档/Notebook/聚合结果的ZIP。它排除原始数据、本机环境、缓存模型和备份。不需要把个人JD汇总表上传。

本轮只准备本机项目与教程，不创建远端账户、不替你填写邮箱或仓库URL。实际GitHub发布需要你使用自己的账户执行上面步骤。公开仓库有许可证：原创代码MIT，数据派生输出保留Criteo条款，详见NOTICE；如果希望改原创代码许可证，可在首次发布前调整。
