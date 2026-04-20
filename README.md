# ASR Pipeline — 多阶段字幕生成

基于 Qwen3-ASR 的多阶段字幕生成流水线，支持强制对齐。可将音频转换为结构化字幕（SRT/ASS 格式），包含断句、CSV 文本修正、卡拉OK高亮效果。同时提供 CLI 和 REST API 两种接口。

## 架构

### 4 阶段流水线

```
阶段 1: ASR + 强制对齐  → 扁平词级 JSON
阶段 2: 断句           → 按段落（。！？）分句，再按 max_chars 分行
阶段 3: CSV 修正       → 应用多个 CSV 修正文件
阶段 4: 渲染           → 输出 SRT 和 ASS，支持样式选择
```

每个阶段都会保存中间 JSON，便于调试和断点续传。

### 后端策略

- **CUDA**: Linux + NVIDIA GPU — 使用 `qwen_asr` + PyTorch 配合 `Qwen3ForcedAligner`
- **MLX**: macOS + Apple Silicon — 使用 `mlx-audio`，ASR 和对齐模型独立加载
- 通过 `asr.platform.get_backend()` 自动检测

### 核心模块（`asr/` 包）

| 模块 | 职责 |
|------|------|
| `asr/pipeline.py` | 流水线编排，阶段协调，断点续传 |
| `asr/engine.py` | Qwen3-ASR 引擎（CUDA/MLX 后端），强制对齐 |
| `asr/subtitle.py` | SRT/ASS 渲染，卡拉OK效果标签（\kf）|
| `asr/text_utils.py` | CJK 字符处理，标点符号，字符计数 |
| `asr/platform.py` | 后端检测（cuda/mlx）|
| `asr/model_path.py` | 模型路径解析与缓存 |
| `asr/download_models.py` | 模型下载脚本 |

### 入口文件

| 模块 | 职责 |
|------|------|
| `main.py` | CLI 入口 |
| `api.py` | FastAPI REST 服务 |

### 数据结构

- `WordTimestamp` — 单词级时间戳（起始/结束时间）
- `ASRResult` — 扁平词列表（ASR 阶段不分组）
- `Paragraph` — 句子级单元，以 。！？ 标点为界
- `SubtitleLine` — 单行字幕，包含时间、词级数据、pause_after
- `ASSSubtitleStyle` — ASS 样式配置（字体、颜色、卡拉OK模式）

### 字符计数

CJK 字符 = 1，非 CJK 字母 = 0.5，标点符号 = 0。用于按 max_chars 断行。

## 安装

包管理使用 [uv](https://github.com/astral-sh/uv)。

```bash
# 安装依赖
uv pip install -e ".[cuda]"  # CUDA 版（Linux + NVIDIA GPU）
uv pip install -e ".[mlx]"   # Apple Silicon 版（macOS M1/M2/M3/M4）
uv pip install -e ".[dev]"   # 开发依赖
```

## CLI 用法

```bash
# 基本转录（输出 SRT）
uv run main.py transcribe audio.wav --output ./subs

# 指定 ASS 格式和样式
uv run main.py transcribe audio.wav --fmt ass --style default

# 指定语言
uv run main.py transcribe audio.wav --language Chinese

# 使用小模型（更快，精度略低）
uv run main.py transcribe audio.wav --model-size 0.6B

# 自定义每行最大字符数
uv run main.py transcribe audio.wav --max-chars 20

# 应用 CSV 文本修正
uv run main.py transcribe audio.wav --fix-dir ./fixes

# 从指定阶段恢复
uv run main.py transcribe audio.wav --resume-from render
```

## API 服务

```bash
# 启动 API 服务
uv run main.py serve --host 0.0.0.0 --port 8000

# 开发模式（自动重载）
uv run main.py serve --reload
```

### API 端点

#### `GET /health` — 健康检查

返回当前后端信息。

**响应示例：**
```json
{
  "status": "ok",
  "backend": "mlx"
}
```

---

#### `POST /transcribe` — 完整流水线

音频文件转字幕，支持 SRT/ASS 格式。

**表单参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `audio` | file | 必填 | 音频文件（wav/mp3/flac/m4a/ogg/wma/aac）|
| `language` | string | null | 语言提示（如 "Chinese"）|
| `model_size` | string | "1.7B" | 模型大小："1.7B" 或 "0.6B" |
| `max_chars` | int | 14 | 每行字幕最大字符数 |
| `fmt` | string | "srt" | 输出格式："srt"、"ass" 或 "all" |
| `ass_style` | string | "default" | ASS 样式名称 |
| `fix_dir` | string | null | CSV 修正文件目录 |

**响应示例：**
```json
{
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "srt_path": "/tmp/asr_550e8400.../subtitle.srt",
  "ass_path": null,
  "status": "completed"
}
```

---

#### `POST /align` — 仅做强制对齐

跳过 ASR 识别，直接对音频和给定文本进行强制对齐，输出字幕。

**表单参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `audio` | file | 必填 | 音频文件 |
| `text` | string | 必填 | 要对齐的文本 |
| `language` | string | null | 语言提示 |
| `max_chars` | int | 14 | 每行字幕最大字符数 |
| `fmt` | string | "srt" | 输出格式："srt"、"ass" 或 "all" |
| `ass_style` | string | "default" | ASS 样式名称 |

---

#### `POST /transcribe/text` — 仅转录文本

快速文本转录，无时间戳，适用于不需要字幕格式的场景。

**表单参数：**

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `audio` | file | 必填 | 音频文件 |
| `language` | string | null | 语言提示 |
| `model_size` | string | "1.7B" | 模型大小 |

**响应示例：**
```json
{
  "text": "这是转录的文本内容。",
  "language": "mlx"
}
```

---

#### `GET /download/{task_id}/{fmt}` — 下载字幕文件

下载指定任务生成的字幕文件。

**路径参数：**

| 参数 | 描述 |
|------|------|
| `task_id` | 转录任务返回的 task_id |
| `fmt` | 文件格式："srt" 或 "ass" |

**注意：** 文件在任务完成后一段时间内有效，请及时下载。

## Python API

```python
from asr.pipeline import run_pipeline

result = run_pipeline(
    audio_path='audio.wav',
    output_dir='./subs',
    fmt='srt',           # 'srt', 'ass', 'all'
    ass_style='default', # ASS 样式名
    fix_dir='./fixes',   # 可选，CSV 修正目录
    language='Chinese',  # 可选，语言提示
    model_size='1.7B',   # '1.7B' 或 '0.6B'
    max_chars=14,        # 每行字幕最大字符数
    resume_from=None,    # 'asr', 'break', 'fix', 'render', None
)
```

## CSV 修正格式

修正 CSV 文件应命名为 `fix_1.csv`、`fix_2.csv` 等，放入修正目录：

```csv
# 格式：原文,替换后文本
# 替换为空则删除该行
错误,正确
要删除的文本,
```

## 关键实现细节

- 流水线支持通过 `resume_from` 参数从任意阶段恢复
- `pipeline` 中的 `split_line_after()` 在特定文本处拆分行，同时保留词级时间
- `fix_dir` 中的 CSV 修正按顺序应用（fix_1.csv, fix_2.csv, ...）
- ASS 卡拉OK默认使用 `\kf`（填充）模式，金色高亮 + 白色暗淡
- CUDA 上超过 5 分钟的音频自动分块为 30 秒段
- 长段使用智能算法拆分，寻找最大时间间隙
