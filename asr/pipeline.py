"""ASR Pipeline - Multi-stage subtitle generation pipeline.

Stage 1: ASR + Forced Alignment → flat word-level JSON (no segment grouping)
Stage 2: Sentence Breaking → two-pass: paragraphs by 。！？, then lines by max_chars
Stage 3: CSV Fix → apply multiple fix CSVs for text correction
Check:  Pre-render validation
Stage 4: Render → SRT and ASS output with style selection

Each stage saves intermediate JSON for debugging and resume support.
"""

import csv
import glob
import json
import os
from dataclasses import dataclass
from typing import Optional

from asr.engine import asr_align, align_only, result_to_dict, ASRResult, ASR_MODELS, ALIGNER_MODELS
from asr.subtitle import SubtitleLine, ASSSubtitleStyle, render_srt_from_lines, render_ass_from_lines
from asr.text_utils import (
    SENTENCE_END, COMMA_LIKE, TRAILING_STRIP,
    word_cjk_len, words_cjk_len, text_cjk_count,
    ends_with_any, text_of_words, is_punctuation,
)
from asr.model_path import is_model_cached, ensure_model
from asr.platform import get_backend
from asr.config import get_model_dir, get_model_size, get_intermediate_dir


# ── Stage 1: ASR + Forced Alignment ──────────────────────────────

def stage1_asr(audio_path: str, output_dir: str, language: Optional[str] = None, model_size: str = "1.7B", task_id: Optional[str] = None) -> dict:
    """Stage 1: Run ASR with forced alignment, save flat word-level JSON.

    Returns dict with 'text', 'language', 'duration', 'words' (flat list).
    No segment grouping.
    """
    print("[Stage 1] ASR + Forced Alignment...")
    result = asr_align(audio_path, language=language, model_size=model_size)
    result_dict = result_to_dict(result)
    result_dict["source"] = os.path.basename(audio_path)

    raw_path = os.path.join(output_dir, _raw_json_name(audio_path, task_id))
    os.makedirs(output_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {raw_path} ({len(result_dict.get('words', []))} words)")
    return result_dict


def stage1_align(audio_path: str, output_dir: str, align_text: str,
                 language: Optional[str] = None, task_id: Optional[str] = None) -> dict:
    """Stage 1 (alignment-only): User provides text, we only timestamp it.

    Skips ASR transcription. Uses ForcedAligner directly on provided text.

    Returns dict with 'text', 'language', 'duration', 'words' (flat list).
    """
    print("[Stage 1] Alignment-only (text provided by user)...")
    result = align_only(audio_path, text=align_text, language=language)
    result_dict = result_to_dict(result)
    result_dict["source"] = os.path.basename(audio_path)

    raw_path = os.path.join(output_dir, _raw_json_name(audio_path, task_id))
    os.makedirs(output_dir, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {raw_path} ({len(result_dict.get('words', []))} words)")
    return result_dict


# ── Stage 2: Sentence Breaking (two-pass) ────────────────────────

def stage2_break(result_dict: dict, output_dir: str, audio_path: str,
                 max_chars: int = 14, task_id: Optional[str] = None) -> list[SubtitleLine]:
    """Stage 2: Text-first sentence breaking.

    Pass 1: Split flat words into paragraphs by sentence-end punctuation (。！？)
    Pass 2: Within each paragraph, break into subtitle lines by max_chars

    Returns list of SubtitleLine.
    """
    print(f"[Stage 2] Sentence Breaking (max_chars={max_chars})...")

    words = result_dict.get("words", [])
    if not words:
        print("  No words found, skipping.")
        return []

    # Pass 1: Build paragraphs
    paragraphs = _build_paragraphs(words)
    print(f"  Pass 1: {len(paragraphs)} paragraphs")

    # Pass 2: Break paragraphs into lines
    all_lines = []
    for para in paragraphs:
        para_lines = _break_paragraph(para, max_chars)
        all_lines.extend(para_lines)

    # Compute pause_after for each line
    for i in range(len(all_lines) - 1):
        gap = all_lines[i + 1].start_time - all_lines[i].end_time
        all_lines[i].pause_after = max(0, gap)

    # Save intermediate
    lines_path = os.path.join(output_dir, _lines_json_name(audio_path, task_id))
    _save_lines(all_lines, lines_path)
    print(f"  Pass 2: {len(all_lines)} lines, saved: {lines_path}")
    return all_lines


def _build_paragraphs(words: list) -> list:
    """Pass 1: Split words into paragraphs at sentence-end punctuation.

    A paragraph ends when a word contains 。！？ or similar.
    The punctuation stays with the paragraph that contains it.
    """
    from asr.subtitle import Paragraph

    paragraphs = []
    current = []

    for word in words:
        current.append(word)
        if ends_with_any(word.get("text", ""), SENTENCE_END):
            # Emit paragraph
            text = text_of_words(current)
            start = current[0].get("start_time", 0)
            end = current[-1].get("end_time", start)
            paragraphs.append(Paragraph(
                text=text, start_time=start, end_time=end,
                words=[dict(w) for w in current],
            ))
            current = []

    # Flush remaining words (no sentence-end punctuation at the end)
    if current:
        text = text_of_words(current)
        start = current[0].get("start_time", 0)
        end = current[-1].get("end_time", start)
        paragraphs.append(Paragraph(
            text=text, start_time=start, end_time=end,
            words=[dict(w) for w in current],
        ))

    return paragraphs


def _break_paragraph(para, max_chars: int) -> list:
    """Pass 2: Break paragraph into subtitle lines at punctuation."""
    words = para.words
    if not words:
        return []

    # Break at every comma/semicolon/colon first
    segments = []
    current_words = []

    for word in words:
        word_text = word.get("text", "")
        current_words.append(word)

        if ends_with_any(word_text, COMMA_LIKE):
            segments.append(list(current_words))
            current_words = []

    # Flush remaining
    if current_words:
        segments.append(list(current_words))

    # Process each segment: if too long, use smart splitting
    lines = []
    for seg_words in segments:
        seg_len = words_cjk_len(seg_words)
        if seg_len <= max_chars:
            _emit_line(lines, seg_words)
        else:
            # Use smart algorithm to split long segment
            sub_lines = _smart_split(seg_words, max_chars)
            lines.extend(sub_lines)

    return lines


def _smart_split(words: list, max_chars: int) -> list:
    """Smart splitting for over-length segments using time gaps.

    Algorithm:
    1. Find all "valid split points" where both sides <= max_chars
    2. Among valid points, pick the one with largest time gap
    3. Recursively apply to left and right parts
    """
    if not words:
        return []

    total_len = words_cjk_len(words)
    if total_len <= max_chars:
        line = _words_to_line(words)
        return [line] if line else []

    # Find all valid split points
    valid_points = _find_valid_split_points(words, max_chars)

    if not valid_points:
        # No valid point found, force split at largest gap in middle region
        split_idx = _find_best_force_split(words, max_chars)
        if split_idx <= 0 or split_idx >= len(words):
            line = _words_to_line(words)
            return [line] if line else []

        return _split_and_recurse(words, split_idx, max_chars)

    # Pick the valid point with largest time gap
    best_idx = max(valid_points, key=lambda i: _get_time_gap(words, i))
    return _split_and_recurse(words, best_idx, max_chars)


def _split_and_recurse(words: list, split_idx: int, max_chars: int) -> list:
    """Split words at index and recursively smart-split both halves."""
    if split_idx <= 0 or split_idx >= len(words):
        line = _words_to_line(words)
        return [line] if line else []
    left_lines = _smart_split(words[:split_idx], max_chars)
    right_lines = _smart_split(words[split_idx:], max_chars)
    return left_lines + right_lines


def _find_valid_split_points(words: list, max_chars: int) -> list:
    """Find all word indices where splitting would make both sides <= max_chars."""
    valid = []
    cum_len = []

    # Build cumulative lengths
    total = 0
    for w in words:
        total += word_cjk_len(w.get("text", ""))
        cum_len.append(total)

    # Find valid split points
    for i in range(1, len(words)):
        left_len = cum_len[i-1]
        right_len = total - left_len
        if left_len <= max_chars and right_len <= max_chars:
            valid.append(i)

    return valid


def _get_time_gap(words: list, split_idx: int) -> float:
    """Get the time gap between words at split point."""
    if split_idx <= 0 or split_idx >= len(words):
        return 0.0
    prev_end = words[split_idx - 1].get("end_time", 0)
    curr_start = words[split_idx].get("start_time", 0)
    return curr_start - prev_end


def _find_best_force_split(words: list, max_chars: int) -> int:
    """Find best split point when no valid point exists.

    Strategy: In the middle region (40%-60% of total length),
    find the point with largest time gap.
    Falls back to simple middle split if all gaps are zero.
    """
    if len(words) < 2:
        return 0

    total_len = words_cjk_len(words)
    target_left = total_len * 0.4
    target_right = total_len * 0.6

    cum_len = 0
    start_idx = end_idx = 0

    for i, w in enumerate(words):
        cum_len += word_cjk_len(w.get("text", ""))
        if cum_len >= target_left and start_idx == 0:
            start_idx = i + 1
        if cum_len >= target_right:
            end_idx = i + 1
            break

    if end_idx == 0:
        end_idx = len(words) // 2
    if start_idx == 0:
        start_idx = max(1, len(words) // 2 - 2)

    end_idx = min(end_idx, len(words) - 1)

    best_idx = start_idx
    best_gap = 0
    for i in range(start_idx, end_idx + 1):
        gap = _get_time_gap(words, i)
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    if best_idx == 0 or best_idx >= len(words):
        best_idx = len(words) // 2

    return best_idx


def _words_to_line(words: list) -> Optional[SubtitleLine]:
    """Convert a list of words to SubtitleLine."""
    if not words:
        return None
    text = text_of_words(words)
    start = words[0].get("start_time", 0)
    end = words[-1].get("end_time", start)
    return SubtitleLine(
        text=text,
        start_time=start,
        end_time=end,
        words=[dict(w) for w in words],
    )


def _emit_line(lines: list, words: list):
    """Create a SubtitleLine from words and append to lines list."""
    line = _words_to_line(words)
    if line:
        lines.append(line)


# ── Stage 3: CSV Fix ─────────────────────────────────────────────

def stage3_fix(lines: list, fix_dir: str) -> list:
    """Stage 3: Apply multiple fix CSVs (fix_1.csv, fix_2.csv, ...) sequentially."""
    print(f"[Stage 3] Applying CSV fixes from {fix_dir}...")

    if not os.path.isdir(fix_dir):
        print(f"  Fix directory not found: {fix_dir}, skipping.")
        return lines

    csv_files = sorted(glob.glob(os.path.join(fix_dir, "fix_*.csv")))
    if not csv_files:
        print("  No fix_*.csv files found, skipping.")
        return lines

    total_replacements = 0
    total_deletions = 0

    for csv_file in csv_files:
        fixes = _load_csv(csv_file)
        print(f"  Applying {os.path.basename(csv_file)} ({len(fixes)} rules)...")

        for line in lines:
            original = line.text
            for orig, repl in fixes.items():
                if orig in line.text:
                    if repl == "":
                        line.text = ""
                        line.words = []
                        total_deletions += 1
                        break
                    else:
                        line.text = line.text.replace(orig, repl)
                        line.words = _rebuild_words(line.words, orig, repl)

            if line.text != original and line.text:
                total_replacements += 1

    before = len(lines)
    lines = [l for l in lines if l.text]
    after = len(lines)
    print(f"  Replacements: {total_replacements}, Deletions: {total_deletions} "
          f"({before} → {after} lines)")
    return lines


def _rebuild_words(words: list, orig: str, repl: str) -> list:
    """Rebuild word list after text replacement, preserving timing for unchanged parts."""
    if not words:
        return []
    result = []
    for w in words:
        text = w.get("text", "")
        if orig in text:
            idx = text.index(orig)
            before_text = text[:idx]
            after_text = text[idx + len(orig):]
            if before_text:
                result.append(dict(w, text=before_text))
            result.append(dict(w, text=repl))
            if after_text:
                result.append(dict(w, text=after_text))
        else:
            result.append(dict(w))
    return result


def _load_csv(csv_path: str) -> dict:
    """Load a fix CSV file. Returns {original: replacement}."""
    fixes = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 2:
                original = row[0].strip()
                new_text = row[1].strip() if len(row) > 1 else ""
                if original and not original.startswith("#"):
                    fixes[original] = new_text
    return fixes


# ── Check ────────────────────────────────────────────────────────

@dataclass
class CheckError:
    """A single check failure."""
    line_idx: int
    checker: str
    message: str
    fix_command: str = ""


def check_max_chars(lines: list, max_chars: int = 14) -> list:
    """Check that no line exceeds max_chars CJK characters."""
    errors = []
    for idx, line in enumerate(lines):
        if not line.text:
            continue
        char_count = text_cjk_count(line.text)
        if char_count > max_chars:
            fix = f"xt asr-split <lines.json> --line {idx + 1} --after \"...\""
            errors.append(CheckError(
                line_idx=idx + 1, checker="max_chars",
                message=f"Line {idx + 1} has {char_count} chars (max {max_chars}): \"{line.text}\"",
                fix_command=fix,
            ))
    return errors


def _find_split_point(text: str, max_chars: int) -> int:
    """Find the best character position to split text."""
    from asr.text_utils import char_weight

    char_pos = 0.0
    best_pos = max_chars
    best_score = 999.0

    for i, ch in enumerate(text):
        char_pos += char_weight(ch)

        distance = abs(char_pos - max_chars)
        is_soft_break = ch in COMMA_LIKE or ch == ' '
        score = distance - (0.5 if is_soft_break else 0)

        if char_pos >= max_chars * 0.6 and score < best_score:
            best_score = score
            best_pos = i + 1
        if char_pos > max_chars * 1.2:
            break
    return best_pos


def stage_check(lines: list, max_chars: int = 14) -> list:
    """Run all checkers. Returns list of CheckError (empty = pass)."""
    print(f"[Check] Running pre-render checks (max_chars={max_chars})...")
    errors = check_max_chars(lines, max_chars)

    if errors:
        print(f"  FAILED: {len(errors)} issue(s) found")
        for err in errors:
            print(f"  ✗ [{err.checker}] {err.message}")
            if err.fix_command:
                print(f"    Fix: {err.fix_command}")
    else:
        print("  PASSED: All checks OK")
    return errors


# ── Pre-render cleanup ────────────────────────────────────────────

def _strip_trailing_punct(lines: list):
    """Remove trailing commas and periods from each subtitle line."""
    for line in lines:
        if not line.text:
            continue
        stripped = line.text.rstrip()
        while stripped and stripped[-1] in TRAILING_STRIP:
            stripped = stripped[:-1].rstrip()
        if stripped != line.text:
            line.text = stripped
            # Also strip from the last word
            if line.words:
                last_w = line.words[-1]
                wtext = last_w.get("text", "")
                while wtext and wtext[-1] in TRAILING_STRIP:
                    wtext = wtext[:-1]
                last_w["text"] = wtext


# ── Line Split ───────────────────────────────────────────────────

def split_line_after(lines: list, line_idx: int, after_text: str) -> list:
    """Split a line after the matched text.

    Finds `after_text` in the line's word stream and breaks after it.
    Raises ValueError if text not found or found multiple times.
    """
    if line_idx < 1 or line_idx > len(lines):
        raise ValueError(f"Line index {line_idx} out of range (1-{len(lines)})")

    target = lines[line_idx - 1]
    words = target.words
    if not words:
        raise ValueError(f"Line {line_idx} has no word-level data, cannot split")

    # Find after_text in concatenated word text
    full_text = text_of_words(words)
    count = full_text.count(after_text)
    if count == 0:
        raise ValueError(f"Text \"{after_text}\" not found in line {line_idx}")
    if count > 1:
        raise ValueError(f"Text \"{after_text}\" found {count} times in line {line_idx}, must be unique")

    pos = full_text.index(after_text)
    end_pos = pos + len(after_text)

    # Find which words cover [0, end_pos)
    cum = 0
    split_word = None
    for i, w in enumerate(words):
        word_text = w.get("text", "")
        word_start = cum
        word_end = cum + len(word_text)
        cum = word_end

        if word_end <= end_pos:
            continue
        if word_start >= end_pos:
            split_word = i
            break
        # end_pos falls inside this word — split the word itself
        if word_start < end_pos < word_end:
            split_offset = end_pos - word_start
            left_text = word_text[:split_offset]
            right_text = word_text[split_offset:]
            # Keep timing on both halves
            words[i] = dict(w)
            words[i]["text"] = left_text
            right_word = dict(w)
            right_word["text"] = right_text
            words.insert(i + 1, right_word)
            split_word = i + 1
            break

    if split_word is None:
        # after_text covers the entire line tail — no split needed
        return lines

    if split_word == 0:
        raise ValueError(f"Split point is at the start of the line, cannot split")

    # Build two new lines
    before_words = words[:split_word]
    after_words = words[split_word:]

    new_lines = []
    if before_words:
        new_lines.append(_make_line(before_words, target))
    if after_words:
        new_lines.append(_make_line(after_words, target))

    return lines[:line_idx - 1] + new_lines + lines[line_idx:]


def _make_line(words: list, template: SubtitleLine) -> SubtitleLine:
    """Create a SubtitleLine from words, using template for fallback timing."""
    return SubtitleLine(
        text=text_of_words(words),
        start_time=words[0].get("start_time", template.start_time),
        end_time=words[-1].get("end_time", template.end_time),
        words=[dict(w) for w in words],
    )


# ── Stage 4: Render ──────────────────────────────────────────────

def stage4_render(lines: list, output_dir: str, audio_path: str,
                  fmt: str = "srt", ass_style="default"):
    """Stage 4: Render SubtitleLines to SRT and/or ASS files."""
    base = os.path.splitext(os.path.basename(audio_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    paths = {}

    if fmt in ("srt", "all"):
        srt_path = os.path.join(output_dir, f"{base}.srt")
        render_srt_from_lines(lines, srt_path)
        paths["srt"] = srt_path
        print(f"  SRT: {srt_path}")

    if fmt in ("ass", "all"):
        ass_path = os.path.join(output_dir, f"{base}.ass")
        style = ASSSubtitleStyle.from_name(ass_style)
        render_ass_from_lines(lines, ass_path, style)
        paths["ass"] = ass_path
        print(f"  ASS: {ass_path}")

    return paths


# ── Pipeline Orchestrator ─────────────────────────────────────────

def _ensure_models(model_size: str) -> bool:
    """Check and download required models if missing."""
    backend = get_backend()
    models = ASR_MODELS.get(backend, ASR_MODELS["cuda"])
    cache_dir = str(get_model_dir())

    if model_size not in models:
        print(f"Error: Unknown model size '{model_size}'. Available: {', '.join(models.keys())}")
        return False

    asr_model_id = models[model_size]
    aligner_model_id = ALIGNER_MODELS[backend]

    print(f"Using backend: {backend}")
    print(f"Model directory: {cache_dir}")

    # Check ASR model
    if not is_model_cached(asr_model_id, cache_dir):
        print(f"ASR model not found, downloading...")
        if not ensure_model(asr_model_id, cache_dir):
            return False

    # Check aligner model
    if not is_model_cached(aligner_model_id, cache_dir):
        print(f"Aligner model not found, downloading...")
        if not ensure_model(aligner_model_id, cache_dir):
            return False

    return True


def run_pipeline(
    audio_path: str,
    output_dir: Optional[str] = None,
    fmt: str = "srt",
    ass_style: str = "default",
    fix_dir: Optional[str] = None,
    language: Optional[str] = None,
    model_size: Optional[str] = None,
    max_chars: int = 14,
    resume_from: Optional[str] = None,
    align_text: Optional[str] = None,
    intermediate_dir: Optional[str] = None,
    task_id: Optional[str] = None,
) -> dict:
    """Run the full ASR subtitle generation pipeline."""
    # Apply defaults
    if model_size is None:
        model_size = get_model_size()
    if model_size not in ("0.6B", "1.7B"):
        model_size = "1.7B"

    # Ensure models are available
    if not _ensure_models(model_size):
        raise RuntimeError(f"Failed to ensure models for backend {get_backend()}")

    audio_path = os.path.abspath(audio_path)
    if output_dir is None:
        output_dir = os.path.dirname(audio_path)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Intermediate products dir: param > config > output_dir
    if intermediate_dir is None:
        intermediate_dir = get_intermediate_dir()
    if intermediate_dir is None:
        intermediate_dir = output_dir
    intermediate_dir = os.path.abspath(intermediate_dir)
    os.makedirs(intermediate_dir, exist_ok=True)

    all_stages = ["asr", "break", "fix", "render"]
    start_idx = 0
    if resume_from:
        if resume_from in all_stages:
            start_idx = all_stages.index(resume_from)
        else:
            print(f"Warning: Unknown resume stage '{resume_from}', starting from beginning.")

    # Stage 1: ASR + Align
    if start_idx <= 0:
        if align_text:
            result_dict = stage1_align(audio_path, intermediate_dir, align_text, language, task_id)
        else:
            result_dict = stage1_asr(audio_path, intermediate_dir, language, model_size, task_id)
    else:
        raw_path = os.path.join(intermediate_dir, _raw_json_name(audio_path, task_id))
        print(f"[Stage 1] Skipping, loading: {raw_path}")
        with open(raw_path, "r", encoding="utf-8") as f:
            result_dict = json.load(f)

    def clean_more_punctuation(text: str) -> str:
        if not text:
            return text
        result = [text[0]]
        for char in text[1:]:
            prev = result[-1]
            if not (is_punctuation(char) and is_punctuation(prev) and char == prev):
                result.append(char)
        return ''.join(result)

    # Clean up duplicate consecutive punctuation in word texts
    for word in result_dict.get("words", []):
        text = word.get("text", "")
        word["text"] = clean_more_punctuation(text)

    # Stage 2: Sentence Breaking
    if start_idx <= 1:
        lines = stage2_break(result_dict, intermediate_dir, audio_path, max_chars, task_id)
    else:
        lines_path = os.path.join(intermediate_dir, _lines_json_name(audio_path, task_id))
        print(f"[Stage 2] Skipping, loading: {lines_path}")
        lines = _load_lines(lines_path)

    # Stage 3: CSV Fix
    if fix_dir and start_idx <= 2:
        lines = stage3_fix(lines, fix_dir)
        lines_path = os.path.join(intermediate_dir, _lines_json_name(audio_path, task_id))
        _save_lines(lines, lines_path)
    elif fix_dir and start_idx > 2:
        lines_path = os.path.join(intermediate_dir, _lines_json_name(audio_path, task_id))
        print(f"[Stage 3] Skipping, loading: {lines_path}")
        lines = _load_lines(lines_path)
    else:
        print("[Stage 3] No fix directory specified, skipping.")

    # Check
    errors = stage_check(lines, max_chars)
    if errors:
        print(f"\n[BLOCKED] Fix {len(errors)} issue(s) before rendering.")
        print("Use 'xt asr-split' to split long lines, then re-run with --resume-from render")
        return {"check_errors": errors, "lines_path": os.path.join(intermediate_dir, _lines_json_name(audio_path, task_id))}

    # Pre-render: strip trailing punctuation from each line
    _strip_trailing_punct(lines)

    # Stage 4: Render
    paths = stage4_render(lines, output_dir, audio_path, fmt, ass_style)
    print(f"[Done] Pipeline complete.")
    return paths


# ── Persistence helpers ───────────────────────────────────────────

def _raw_json_name(audio_path: str, task_id: Optional[str] = None) -> str:
    if task_id:
        return f"{task_id}.raw.json"
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return f"{base}.raw.json"


def _lines_json_name(audio_path: str, task_id: Optional[str] = None) -> str:
    if task_id:
        return f"{task_id}.lines.json"
    base = os.path.splitext(os.path.basename(audio_path))[0]
    return f"{base}.lines.json"


def _save_lines(lines: list, path: str):
    data = [_line_to_dict(l) for l in lines]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_lines(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SubtitleLine(**d) for d in data]


def _line_to_dict(line: SubtitleLine) -> dict:
    return {
        "text": line.text,
        "start_time": line.start_time,
        "end_time": line.end_time,
        "words": line.words,
        "pause_after": line.pause_after,
    }
