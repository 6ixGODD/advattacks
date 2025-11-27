from __future__ import annotations

import enum
import os
import sys


class ANSI:
    class FG(enum.StrEnum):
        BLACK = "\033[30m"
        RED = "\033[31m"
        GREEN = "\033[32m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"
        MAGENTA = "\033[35m"
        CYAN = "\033[36m"
        WHITE = "\033[37m"
        GRAY = "\033[90m"
        BRIGHT_RED = "\033[91m"
        BRIGHT_GREEN = "\033[92m"
        BRIGHT_YELLOW = "\033[93m"
        BRIGHT_BLUE = "\033[94m"
        BRIGHT_MAGENTA = "\033[95m"
        BRIGHT_CYAN = "\033[96m"
        BRIGHT_WHITE = "\033[97m"

    class BG(enum.StrEnum):
        BLACK = "\033[40m"
        RED = "\033[41m"
        GREEN = "\033[42m"
        YELLOW = "\033[43m"
        BLUE = "\033[44m"
        MAGENTA = "\033[45m"
        CYAN = "\033[46m"
        WHITE = "\033[47m"
        GRAY = "\033[100m"
        BRIGHT_RED = "\033[101m"
        BRIGHT_GREEN = "\033[102m"
        BRIGHT_YELLOW = "\033[103m"
        BRIGHT_BLUE = "\033[104m"
        BRIGHT_MAGENTA = "\033[105m"
        BRIGHT_CYAN = "\033[106m"
        BRIGHT_WHITE = "\033[107m"

    class STYLE(enum.StrEnum):
        RESET = "\033[0m"
        BOLD = "\033[1m"
        DIM = "\033[2m"
        ITALIC = "\033[3m"
        UNDERLINE = "\033[4m"
        BLINK = "\033[5m"
        REVERSE = "\033[7m"
        HIDDEN = "\033[8m"
        STRIKETHROUGH = "\033[9m"

    # For backward compatibility
    RESET = STYLE.RESET
    BOLD = STYLE.BOLD
    UNDERLINE = STYLE.UNDERLINE
    REVERSED = STYLE.REVERSE
    RED = FG.BRIGHT_RED
    GREEN = FG.BRIGHT_GREEN
    YELLOW = FG.BRIGHT_YELLOW
    BLUE = FG.BRIGHT_BLUE
    MAGENTA = FG.BRIGHT_MAGENTA
    CYAN = FG.BRIGHT_CYAN
    WHITE = FG.BRIGHT_WHITE

    _enabled = True

    @classmethod
    def supports_color(cls) -> bool:
        """Determine if the current terminal supports colors."""
        if os.environ.get("NO_COLOR", ""):
            return False
        if os.environ.get("FORCE_COLOR", ""):
            return True
        return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    @classmethod
    def enable(cls, enabled: bool = True) -> None:
        """Enable or disable ANSI formatting."""
        cls._enabled = enabled and cls.supports_color()

    @classmethod
    def format(cls, text: str, /, *styles) -> str:
        """Format text with the specified ANSI styles."""
        if not cls._enabled or not styles:
            return text

        valid_styles = [
            str(s.value) if hasattr(s, "value") else str(s) for s in styles if s is not None
        ]

        if not valid_styles:
            return text

        style_str = "".join(valid_styles)

        if cls.STYLE.RESET in text:
            text = text.replace(cls.STYLE.RESET, f"{cls.STYLE.RESET}{style_str}")

        return f"{style_str}{text}{cls.STYLE.RESET}"

    @classmethod
    def success(cls, text: str, /) -> str:
        """Format text as a success message (green, bold)."""
        return cls.format(text, cls.FG.BRIGHT_GREEN, cls.STYLE.BOLD)

    @classmethod
    def error(cls, text: str, /) -> str:
        """Format text as an error message (red, bold)."""
        return cls.format(text, cls.FG.BRIGHT_RED, cls.STYLE.BOLD)

    @classmethod
    def warning(cls, text: str, /) -> str:
        """Format text as a warning message (yellow, bold)."""
        return cls.format(text, cls.FG.BRIGHT_YELLOW, cls.STYLE.BOLD)

    @classmethod
    def info(cls, text: str, /) -> str:
        """Format text as an info message (cyan)."""
        return cls.format(text, cls.FG.BRIGHT_CYAN)

    @classmethod
    def highlight(cls, text: str, /) -> str:
        """Format text as highlighted (magenta, bold)."""
        return cls.format(text, cls.FG.BRIGHT_MAGENTA, cls.STYLE.BOLD)


ANSI.enable()
