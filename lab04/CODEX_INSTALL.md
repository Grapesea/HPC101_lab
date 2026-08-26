# 将 Codex 安装到用户主目录的子目录

可以把 Codex 的可执行文件和配置数据都放到 `~/apps/codex/` 下。建议使用官方独立安装脚本，不需要 root 权限。

## 安装

```bash
mkdir -p "$HOME/apps/codex/bin" "$HOME/apps/codex/home"

curl -fsSL https://chatgpt.com/codex/install.sh |
  CODEX_INSTALL_DIR="$HOME/apps/codex/bin" \
  CODEX_HOME="$HOME/apps/codex/home" \
  sh
```

## 配置环境变量

把 Codex 的目录加入 `PATH`，并永久设置 `CODEX_HOME`：

```bash
printf '\nexport CODEX_HOME="$HOME/apps/codex/home"\nexport PATH="$HOME/apps/codex/bin:$PATH"\n' >> "$HOME/.bashrc"
source "$HOME/.bashrc"
```

## 检查安装并登录

```bash
command -v codex
codex --version
codex login
```

## 在项目中运行

```bash
cd ~/HPC101_lab/lab4
codex
```

## 注意当前用户

如果终端提示符是 `root@...`，那么当前的 `~` 指向 `/root`。如果希望将 Codex 安装到普通用户的主目录，应先切换到该用户，例如：

```bash
su - h3240104505
```

然后再执行前面的安装命令。

## 目录变量说明

- `CODEX_INSTALL_DIR`：控制 `codex` 命令的安装位置。
- `CODEX_HOME`：控制配置、登录信息、日志、会话和技能等数据的位置；该目录需要预先创建。

参考：[Codex 环境变量说明](https://learn.chatgpt.com/docs/config-file/environment-variables)
