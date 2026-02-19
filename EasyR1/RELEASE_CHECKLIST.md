# EasyR1 Release 前检查清单

本文档列出了在发布前需要修复或注意的问题。

## 🔴 必须修复（不适合 release）

### 1. 硬编码的私有 IP 和 API 配置
**文件**: `examples/reward_function/r1gui_grounders.py`
- **第 12 行**: `grounding_urls = ['http://10.42.77.111:8000/v1']` — 硬编码了内网 IP，应改为环境变量或配置
- **第 19 行**: `api_key='token-abc123'` — 建议改为从环境变量读取

**建议**: 使用 `os.getenv("GROUNDING_URLS", "http://127.0.0.1:8000/v1")` 或配置文件

### 2. 个人开发路径和调试脚本
**文件**: `analysis_eval.py`
- **第 4 行**: `path = f'/data/users/ruiyang/EasyR1/checkpoints/...'` — 硬编码了个人路径
- **用途**: 看起来是个人分析/调试脚本，不适合作为项目的一部分发布

**建议**: 删除或移到项目外；若保留，应使用命令行参数传入路径

### 3. upload_model.py 中的大量注释路径
**文件**: `upload_model.py`
- **第 6-50+ 行**: 大量注释掉的个人 checkpoint 路径（`/data/users/ruiyang/`, `/data/datafromoldb200/` 等）
- 暴露了内部目录结构和开发环境

**建议**: 清理这些注释，或改为使用命令行参数

### 4. setup.py 中的 URL 指向错误仓库
**文件**: `setup.py`
- **第 51 行**: `url="https://github.com/volcengine/verl"` — 指向原 veRL 仓库，而非 EasyR1

**建议**: 改为 `url="https://github.com/hiyouga/EasyR1"`

---

## 🟡 建议改进（影响体验）

### 5. .gitignore 缺少 .vscode
- **现状**: `.vscode/` 未被忽略，git status 显示为未跟踪
- **建议**: 在 `.gitignore` 中添加 `.vscode/`

### 6. 调试用 print 语句
多处存在 `random.random() < 0.02` 触发的调试 print，生产环境会产生噪音：
- `verl/trainer/core_algos.py`: 第 569, 617 行
- `examples/reward_function/r1gui_grounders.py`: 第 414-420 行 (1% 概率打印 Debug Info)

**建议**: 改为使用 `logging` 模块，并通过 log level 控制

### 7. 示例脚本的调试用途说明
**文件**: `examples/qwen2_5_vl_7b_multi_image.sh`
- 第 2 行已注明: "ONLY be used for debugging. DO NOT use for training"
- 建议在 examples 目录下单独标注或移出主示例

### 8. examples 中 reward function 的 print 调试输出
多个 `r1gui_*.py` 文件包含 `print(reward_inputs["response"], ...)` 等调试输出，在训练时可能刷屏

---

## 🟢 已知限制（README 已说明）

- **TODO**: LoRA 支持、ulysses 并行等
- **Known bugs**: VLM 与 ulysses 并行不兼容
- 版本号: `0.3.3.dev0` (开发版)

---

## Git 重置为新仓库

如需完全重置并新建仓库，执行：

```bash
cd /Users/yangrui/Desktop/EasyR1/EasyR1

# 1. 删除现有 .git（会丢失所有提交历史）
rm -rf .git

# 2. 重新初始化
git init

# 3. 添加所有文件（.gitignore 会生效）
git add .

# 4. 首次提交
git commit -m "Initial commit"
```

**注意**: 若需保留原仓库的 remote 配置，在 `rm -rf .git` 前可先备份：
```bash
# 备份 remote 信息
git remote -v > /tmp/remote_backup.txt
```
