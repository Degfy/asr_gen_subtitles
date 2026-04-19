r"""ASS Karaoke Subtitle Generator.

Generates SRT and ASS subtitle files from subtitle line data.
Uses \kf karaoke effect with white text + black border, golden highlight.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional

from asr.text_utils import is_cjk, is_punctuation, char_weight

# ── Time formatting ──

def format_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def format_srt_time(seconds: float) -> str:
    """Convert seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


# ── Text helpers ──

def _split_punctuation(text: str) -> tuple[List[str], str]:
    """Split text into core characters and trailing punctuation."""
    core: list[str] = []
    trail = ""
    for ch in text:
        if not is_punctuation(ch):
            core.append(ch)
        elif core:
            trail += ch
    return core, trail


# ── ASS Subtitle Style ────────────────────────────────────────────

@dataclass
class ASSSubtitleStyle:
    """ASS subtitle style configuration.

    Default: White text with black border, golden highlight for karaoke.
    """

    # ── Character (字符) ──
    font: str = "Source Han Sans SC"
    bold: bool = True
    italic: bool = False
    size: int = 50
    color: str = "&H00FFFFFF"          # White (BGR)
    spacing: float = 0

    # ── Border (边框) ──
    border_width: float = 3
    border_color: str = "&H00000000"   # Black

    # ── Shadow (投影) ──
    shadow_enabled: bool = True
    shadow_color: str = "&H00000000"   # Black
    shadow_offset_x: float = 2
    shadow_offset_y: float = 2
    shadow_blur: float = 0
    shadow_opacity: int = 80           # 0=transparent, 100=opaque

    # ── Position (位置) ──
    alignment: int = 8                 # 8=top-center
    margin_v: int = 150
    margin_l: int = 10
    margin_r: int = 10

    # ── Highlight / Karaoke (高亮) ──
    highlight_enabled: bool = True
    highlight_mode: str = "fill"         # "fill" = \kf karaoke
    highlight_color: str = "&H0000D7FF"  # Golden (BGR)
    highlight_scale: float = 108
    dim_color: str = "&H00FFFFFF"        # White (unhighlighted)

    # ── Meta ──
    style_name: str = "Karaoke"

    @property
    def has_asymmetric_shadow(self) -> bool:
        r"""Whether shadow offset is asymmetric (needs inline \xshad/\yshad tags)."""
        return self.shadow_offset_x != self.shadow_offset_y

    def _shadow_alpha_byte(self) -> int:
        """Shadow alpha: 0=opaque, 255=transparent."""
        return round((100 - self.shadow_opacity) / 100 * 255)

    def _back_colour_hex(self) -> str:
        """BackColour (shadow color with alpha) for Style line."""
        if not self.shadow_enabled:
            return "&H80000000"
        bgr = self.shadow_color[4:]  # extract BGR from &HAABBGGRR
        return f"&H{self._shadow_alpha_byte():02X}{bgr}"

    @property
    def primary_colour(self) -> str:
        """PrimaryColour for Style line (filled/highlighted color)."""
        if not self.highlight_enabled:
            return self.color
        if self.highlight_mode == "fill":
            return self.highlight_color  # golden for karaoke fill
        return self.dim_color

    @property
    def secondary_colour(self) -> str:
        """SecondaryColour for Style line (unfilled color in karaoke)."""
        return self.color  # white

    def to_style_line(self) -> str:
        """Generate ASS Style definition line."""
        shadow_depth = self.shadow_offset_x if (
            self.shadow_enabled and not self.has_asymmetric_shadow
        ) else 0
        return (
            f"Style: {self.style_name},{self.font},{self.size},"
            f"{self.primary_colour},{self.secondary_colour},"
            f"{self.border_color},{self._back_colour_hex()},"
            f"{'-1' if self.bold else '0'},{'-1' if self.italic else '0'},0,0,"
            f"100,100,{self.spacing:.0f},0,"
            f"1,{self.border_width:.0f},{shadow_depth:.0f},"
            f"{self.alignment},"
            f"{self.margin_l},{self.margin_r},{self.margin_v},1"
        )

    def shadow_tags(self) -> str:
        """Inline override tags for shadow (per Dialogue line, needed for asymmetric)."""
        if not self.shadow_enabled:
            return ""
        parts: list[str] = []
        if self.has_asymmetric_shadow:
            alpha_hex = f"&H{self._shadow_alpha_byte():02X}&"
            parts.append(f"\\4c{self.shadow_color}")
            parts.append(f"\\4a{alpha_hex}")
            parts.append(f"\\xshad{self.shadow_offset_x:.0f}")
            parts.append(f"\\yshad{self.shadow_offset_y:.0f}")
        if self.shadow_blur > 0:
            parts.append(f"\\be{self.shadow_blur:.0f}")
        return "{" + "".join(parts) + "}" if parts else ""

    def to_header(self) -> str:
        """Generate complete ASS header section."""
        return (
            "[Script Info]\n"
            "Title: Karaoke Subtitles\n"
            "ScriptType: v4.00+\n"
            "PlayResX: 1920\n"
            "PlayResY: 1080\n"
            "WrapStyle: 0\n"
            "ScaledBorderAndShadow: yes\n"
            "[V4+ Styles]\n"
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"{self.to_style_line()}\n\n"
            "[Events]\n"
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
        )

    @classmethod
    def from_name(cls, name: str) -> "ASSSubtitleStyle":
        """Look up a preset style by name. Only 'default' is available."""
        if name == "default":
            return cls()
        raise ValueError(f"Unknown style '{name}'. Available: default")


# ── Tag builders ─────────────────────────────────────────────────

def build_kf_tags(words: List[Dict]) -> str:
    r"""Build karaoke fill tags (\kf) for smooth left-to-right fill effect.

    Text starts at SecondaryColour (white), fills to PrimaryColour (golden)
    as each character/syllable is spoken. Classic karaoke look.
    """
    tags: list[str] = []

    for word in words:
        text = word.get("text", "")
        start = word.get("start_time", 0)
        end = word.get("end_time", start)

        core_chars, trail_punct = _split_punctuation(text)

        if not core_chars:
            if tags and trail_punct:
                tags[-1] += trail_punct
            continue

        duration_cs = int((end - start) * 100)
        if duration_cs <= 0:
            duration_cs = 10

        if core_chars and is_cjk(core_chars[0]):
            # Character-level karaoke for CJK
            char_cs = duration_cs // len(core_chars)
            for i, ch in enumerate(core_chars):
                suffix = trail_punct if i == len(core_chars) - 1 else ""
                tags.append(f"{{\\kf{char_cs}}}{ch}{suffix}")
        else:
            # Word-level karaoke for non-CJK
            core_text = "".join(core_chars)
            tags.append(f"{{\\kf{duration_cs}}}{core_text}{trail_punct}")

    return "".join(tags)


def build_pulse_tags(words: List[Dict], dialogue_start: float, style: ASSSubtitleStyle) -> str:
    r"""Build pulse-highlight tags using ASS \t() transitions.

    Each character lights up instantly when spoken and dims when done.
    Returns plain text if style.highlight_enabled is False.
    """
    if not style.highlight_enabled:
        return "".join(word.get("text", "") for word in words)

    highlight = style.highlight_color
    dim = style.dim_color
    scale = int(style.highlight_scale)

    tags: list[str] = []

    for word in words:
        text = word.get("text", "")
        start = word.get("start_time", 0)
        end = word.get("end_time", start)

        core_chars, trail_punct = _split_punctuation(text)

        if not core_chars:
            if tags and trail_punct:
                tags[-1] += trail_punct
            continue

        duration = end - start
        if duration <= 0:
            duration = 0.1

        if core_chars and is_cjk(core_chars[0]):
            char_dur = duration / len(core_chars)
            for i, ch in enumerate(core_chars):
                suffix = trail_punct if i == len(core_chars) - 1 else ""
                cs = start + i * char_dur
                ce = start + (i + 1) * char_dur
                t_on = int((cs - dialogue_start) * 1000)
                t_off = int((ce - dialogue_start) * 1000)
                tags.append(
                    f"{{\\t({t_on},{t_on + 1},\\1c{highlight}\\fscx{scale}\\fscy{scale})"
                    f"\\t({t_off},{t_off + 1},\\1c{dim}\\fscx100\\fscy100)}}"
                    f"{ch}{suffix}"
                )
        else:
            t_on = int((start - dialogue_start) * 1000)
            t_off = int((end - dialogue_start) * 1000)
            core_text = "".join(core_chars)
            tags.append(
                f"{{\\t({t_on},{t_on + 1},\\1c{highlight}\\fscx{scale}\\fscy{scale})"
                f"\\t({t_off},{t_off + 1},\\1c{dim}\\fscx100\\fscy100)}}"
                f"{core_text}{trail_punct}"
            )

    return "".join(tags)


# ── Rendering ────────────────────────────────────────────────────

def render_srt_from_lines(lines: list, output_path: str) -> None:
    """Render subtitle lines to SRT file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(lines, 1):
            if not line.text:
                continue
            f.write(f"{i}\n")
            f.write(f"{format_srt_time(line.start_time)} --> {format_srt_time(line.end_time)}\n")
            f.write(f"{line.text}\n\n")


def render_ass_from_lines(lines: list, output_path: str, style: ASSSubtitleStyle) -> None:
    """Render subtitle lines to ASS file with karaoke effect."""
    header = style.to_header()

    dialogues: list[str] = []
    for line in lines:
        if not line.text:
            continue

        start = format_ass_time(line.start_time)
        end = format_ass_time(line.end_time)

        if style.highlight_enabled and style.highlight_mode == "fill" and line.words:
            # Use karaoke fill effect
            effect = "{\\kf1}"
            text = build_kf_tags(line.words)
            text = effect + text + style.shadow_tags()
        elif style.highlight_enabled and line.words:
            # Use pulse effect
            text = build_pulse_tags(line.words, line.start_time, style) + style.shadow_tags()
        else:
            text = line.text + style.shadow_tags()

        dialogues.append(
            f"Dialogue: 0,{start},{end},Karaoke,,0,0,0,,{text}"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write("\n".join(dialogues))
        f.write("\n")


# ── Data structures for pipeline ──────────────────────────────────

@dataclass
class Paragraph:
    """A sentence-level unit, bounded by 。！？ punctuation."""
    text: str
    start_time: float
    end_time: float
    words: list


@dataclass
class SubtitleLine:
    """A single subtitle line with timing and word-level data."""
    text: str
    start_time: float
    end_time: float
    words: list = None
    pause_after: float = 0.0

    def __post_init__(self):
        if self.words is None:
            self.words = []
