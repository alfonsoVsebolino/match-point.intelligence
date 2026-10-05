"""MatchPoint.intelligence dedicated Tkinter desktop application."""

import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
from atp_data import SPLIT_CHOICES, get_tournaments_for_split, get_matches_for_tournament
from atp_engine import DEFAULT_P1, DEFAULT_P2, active_players, all_players
import atp_service

# ---------------------------------------------------------------------------
# Constants & Choices
# ---------------------------------------------------------------------------
SURFACE_CHOICES = ["Hard", "Clay", "Grass", "Carpet"]
SERIES_CHOICES = ["Grand Slam", "Masters 1000", "ATP500", "ATP250"]
DEFAULT_SURFACE = "Hard"
DEFAULT_SERIES = "Grand Slam"

PALETTE = {
    "bg": "#0b0f19",
    "card_bg": "#0f172a",
    "border": "#1e293b",
    "input_bg": "#0b0f19",
    "input_border": "#334155",
    "accent": "#14b8a6",
    "accent_hover": "#0d9488",
    "text": "#f8fafc",
    "text_muted": "#94a3b8",
    "text_dim": "#64748b",
    "swap_btn_bg": "#1e293b",
    "swap_btn_text": "#38bdf8",
    "error_text": "#f43f5e",
    "error_bg": "#1e293b",
    "bar_lgb": "#14b8a6",
    "bar_lr": "#38bdf8",
    "bar_empty": "#334155",
    "correct_bg": "#064e3b",
    "correct_border": "#10b981",
    "correct_text": "#10b981",
    "upset_bg": "#4c0519",
    "upset_border": "#f43f5e",
    "upset_text": "#f43f5e",
}

SURFACE_COLORS = {
    "Hard": "#38bdf8",
    "Clay": "#fb923c",
    "Grass": "#4ade80",
    "Carpet": "#a78bfa",
}

# ---------------------------------------------------------------------------
# Pure Logic Functions
# ---------------------------------------------------------------------------
def resolve_split_change(new_split: str) -> tuple[list[str], str, list[tuple[str, str]], str]:
    """Resolves tournaments and default match selections on split change."""
    if new_split not in SPLIT_CHOICES:
        return [], "", [], ""
    tourns = get_tournaments_for_split(new_split)
    first_tourn = tourns[0] if tourns else ""
    matches = get_matches_for_tournament(new_split, first_tourn) if first_tourn else []
    first_match_id = matches[0][1] if matches else ""
    return tourns, first_tourn, matches, first_match_id

def resolve_tournament_change(cur_split: str, tourn_name: str) -> tuple[list[tuple[str, str]], str]:
    """Resolves matches and default match id on tournament change."""
    if cur_split not in SPLIT_CHOICES or not tourn_name:
        return [], ""
    matches = get_matches_for_tournament(cur_split, tourn_name)
    first_match_id = matches[0][1] if matches else ""
    return matches, first_match_id

def evaluate_historical_outcome(p1: str, actual_winner: str, prob_lgb: float) -> dict:
    """Evaluates whether LGBM favourite matched actual winner and returns badge styling."""
    model_favors_p1 = float(prob_lgb) >= 0.5
    actual_p1_won = (actual_winner == p1)
    correct = (model_favors_p1 == actual_p1_won)
    return {
        "correct": correct,
        "badge_text": "PREDICTION ACCURATE" if correct else "UPSET / DIVERGENT",
        "badge_bg": PALETTE["correct_bg"] if correct else PALETTE["upset_bg"],
        "badge_border": PALETTE["correct_border"] if correct else PALETTE["upset_border"],
        "badge_text_color": PALETTE["correct_text"] if correct else PALETTE["upset_text"],
        "icon": "[OK]" if correct else "[X]",
    }

def format_bm_odds(bm_prob: float | None) -> dict[str, str] | None:
    """Formats market implied bookmaker odds into percentage strings or None."""
    if bm_prob is None:
        return None
    bm_p1 = float(bm_prob)
    bm_p2 = float(1.0 - bm_prob)
    return {
        "bm_p1_pct": f"{bm_p1 * 100:.1f}%",
        "bm_p2_pct": f"{bm_p2 * 100:.1f}%",
    }
def resolve_swap(cur_p1: str, cur_p2: str) -> tuple[str, str]:
    """Exchanges current player 1 and player 2 selections."""
    return cur_p2, cur_p1

def resolve_roster_toggle(
    unlock_all: bool,
    cur_p1: str,
    cur_p2: str,
    active_pool: list[str] | None = None,
    all_pool: list[str] | None = None,
) -> tuple[list[str], str, str]:
    """Resolves roster pool choices and ensures player selections remain valid."""
    act = active_players if active_pool is None else active_pool
    all_p = all_players if all_pool is None else all_pool
    pool = all_p if unlock_all else act

    new_p1 = cur_p1 if cur_p1 in pool else (pool[0] if pool else "")
    new_p2 = cur_p2 if cur_p2 in pool else (pool[1] if len(pool) > 1 else (pool[0] if pool else ""))
    return pool, new_p1, new_p2

def format_error_text(result_or_error) -> str:
    """Extracts a human-readable error string from service dict or string."""
    if isinstance(result_or_error, dict):
        return result_or_error.get("error", "")
    if isinstance(result_or_error, str):
        return result_or_error
    return ""

def compute_display_probabilities(prob_lgb: float, prob_lr: float) -> dict:
    """Computes complementary probabilities and formatted percentage strings."""
    lgb_p1 = float(prob_lgb)
    lgb_p2 = float(1.0 - prob_lgb)
    lr_p1 = float(prob_lr)
    lr_p2 = float(1.0 - prob_lr)
    return {
        "lgb_p1": lgb_p1,
        "lgb_p2": lgb_p2,
        "lgb_p1_pct": f"{lgb_p1 * 100:.1f}%",
        "lgb_p2_pct": f"{lgb_p2 * 100:.1f}%",
        "lr_p1": lr_p1,
        "lr_p2": lr_p2,
        "lr_p1_pct": f"{lr_p1 * 100:.1f}%",
        "lr_p2_pct": f"{lr_p2 * 100:.1f}%",
    }

def resolve_surface_color(surface: str) -> str:
    """Returns the accent color for a given court surface."""
    return SURFACE_COLORS.get(surface, PALETTE["accent"])

def format_symmetry_text(raw_asym: float, sym_score: float = 0.0) -> str:
    """Formats symmetry and raw asymmetry display string."""
    err_str = "Δ < 1e-5" if sym_score < 1e-5 else f"Δ {sym_score:.1e}"
    return f"Symmetry: {err_str} (Raw Δ: {raw_asym * 100:.1f}%)"

def resolve_divergence_status(is_divergent: bool = False, div_delta: float = 0.0) -> tuple[bool, str]:
    """Returns (visible, warning_message) based on model divergence delta threshold (0.15)."""
    visible = bool(is_divergent or div_delta > 0.15)
    pct = div_delta * 100
    msg = f"Non-linear divergence detected (|ΔP| = {pct:.1f}%): LightGBM captures non-linear interactions unmodeled by regularized Logistic Regression."
    return visible, msg

def format_key_stats(key_stats: dict | None) -> list[dict]:
    """Formats 5 key delta summary tiles oriented vs Player 1."""
    stats = key_stats or {}
    rank_diff = float(stats.get("rank_diff", 0.0))
    swr_diff = float(stats.get("surface_winrate_diff", 0.0))
    form_diff = float(stats.get("form_divergence_diff", 0.0))
    fatigue_diff = float(stats.get("matches_14d_diff", 0.0))
    p1_wins = int(stats.get("h2h_p1_wins", 0))
    p2_wins = int(stats.get("h2h_p2_wins", 0))

    rank_color = "#10b981" if rank_diff <= 0 else "#ef4444"
    swr_color = "#10b981" if swr_diff >= 0 else "#ef4444"
    form_color = "#10b981" if form_diff >= 0 else "#ef4444"
    fatigue_color = "#e2e8f0"
    h2h_color = "#10b981" if p1_wins > p2_wins else ("#ef4444" if p1_wins < p2_wins else "#38bdf8")

    return [
        {"id": "rank", "label": "Rank Δ", "val": f"{rank_diff:+.0f}", "color": rank_color, "raw": rank_diff},
        {"id": "swr", "label": "Surface WR Δ", "val": f"{swr_diff * 100:+.1f}%", "color": swr_color, "raw": swr_diff},
        {"id": "form", "label": "Form Div Δ", "val": f"{form_diff:+.2f}", "color": form_color, "raw": form_diff},
        {"id": "fatigue", "label": "Fatigue 14d Δ", "val": f"{fatigue_diff:+.0f} m", "color": fatigue_color, "raw": fatigue_diff},
        {"id": "h2h", "label": "H2H Record", "val": f"{p1_wins} - {p2_wins}", "color": h2h_color, "p1_wins": p1_wins, "p2_wins": p2_wins},
    ]

def resolve_font_family(root: tk.Tk, font_type: str = "sans") -> str:
    """Finds best matching font family on system with fallbacks."""
    try:
        import tkinter.font as tkfont
        available = {f.lower() for f in tkfont.families(root)}
    except Exception:
        available = set()

    if font_type == "mono":
        for cand in ["courier", "nimbus mono l", "consolas", "jetbrains mono"]:
            if cand in available:
                return cand
        return "courier"
    else:
        for cand in ["helvetica", "nimbus sans l", "segoe ui", "plus jakarta sans", "sans-serif"]:
            if cand in available:
                return cand
        return "helvetica"

# ---------------------------------------------------------------------------
# UI Widgets
# ---------------------------------------------------------------------------
class ProbabilityBarsCanvas(tk.Canvas):
    """Canvas drawing dual-model win probability bars."""

    def __init__(self, parent, font_sans: str = "helvetica", font_mono: str = "courier", **kwargs):
        super().__init__(
            parent,
            height=106,
            bg=PALETTE["card_bg"],
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.font_sans = font_sans
        self.font_mono = font_mono
        self.p1 = ""
        self.p2 = ""
        self.prob_lgb = 0.5
        self.prob_lr = 0.5
        self.has_data = False
        self.bind("<Configure>", self._on_configure)

    def set_data(self, p1: str, p2: str, prob_lgb: float, prob_lr: float):
        self.p1 = p1
        self.p2 = p2
        self.prob_lgb = prob_lgb
        self.prob_lr = prob_lr
        self.has_data = True
        self.redraw()

    def _on_configure(self, event=None):
        if self.has_data:
            self.redraw()

    def redraw(self):
        self.delete("all")
        if not self.has_data:
            return

        w = self.winfo_width()
        if w <= 20:
            w = 580

        pad_x = 4
        bar_w = max(20, w - 2 * pad_x)
        probs = compute_display_probabilities(self.prob_lgb, self.prob_lr)

        # LightGBM Champion
        y_lgb_txt = 12
        y_lgb_top = 26
        y_lgb_bot = 40

        self.create_text(
            pad_x, y_lgb_txt,
            text="LightGBM Champion",
            fill=PALETTE["accent"],
            anchor="w",
            font=(self.font_sans, 10, "bold")
        )
        lgb_summary = f"{self.p1}: {probs['lgb_p1_pct']}   |   {self.p2}: {probs['lgb_p2_pct']}"
        self.create_text(
            w - pad_x, y_lgb_txt,
            text=lgb_summary,
            fill=PALETTE["text"],
            anchor="e",
            font=(self.font_mono, 9)
        )

        w_p1_lgb = max(0, min(bar_w, int(bar_w * self.prob_lgb)))
        self.create_rectangle(
            pad_x, y_lgb_top,
            pad_x + bar_w, y_lgb_bot,
            fill=PALETTE["input_border"],
            outline=""
        )
        if w_p1_lgb > 0:
            self.create_rectangle(
                pad_x, y_lgb_top,
                pad_x + w_p1_lgb, y_lgb_bot,
                fill=PALETTE["accent"],
                outline=""
            )

        # Logistic Regression Baseline
        y_lr_txt = 60
        y_lr_top = 74
        y_lr_bot = 86

        self.create_text(
            pad_x, y_lr_txt,
            text="Logistic Regression Baseline",
            fill="#60a5fa",
            anchor="w",
            font=(self.font_sans, 9, "bold")
        )
        lr_summary = f"{self.p1}: {probs['lr_p1_pct']}   |   {self.p2}: {probs['lr_p2_pct']}"
        self.create_text(
            w - pad_x, y_lr_txt,
            text=lr_summary,
            fill=PALETTE["text_muted"],
            anchor="e",
            font=(self.font_mono, 9)
        )

        w_p1_lr = max(0, min(bar_w, int(bar_w * self.prob_lr)))
        self.create_rectangle(
            pad_x, y_lr_top,
            pad_x + bar_w, y_lr_bot,
            fill=PALETTE["input_border"],
            outline=""
        )
        if w_p1_lr > 0:
            self.create_rectangle(
                pad_x, y_lr_top,
                pad_x + w_p1_lr, y_lr_bot,
                fill=PALETTE["bar_lr"],
                outline=""
            )


class BookmakerBarCanvas(tk.Canvas):
    """Canvas drawing horizontal market implied probability bar."""

    def __init__(self, parent, **kwargs):
        super().__init__(
            parent,
            height=6,
            bg=PALETTE["card_bg"],
            highlightthickness=0,
            bd=0,
            **kwargs
        )
        self.bm_prob = 0.5
        self.has_data = False
        self.bind("<Configure>", self._on_configure)

    def set_prob(self, bm_prob: float):
        self.bm_prob = max(0.0, min(1.0, float(bm_prob)))
        self.has_data = True
        self.redraw()

    def _on_configure(self, event=None):
        if self.has_data:
            self.redraw()

    def redraw(self):
        self.delete("all")
        if not self.has_data:
            return
        w = self.winfo_width()
        if w <= 20:
            w = 580
        pad_x = 4
        bar_w = max(20, w - 2 * pad_x)
        w1 = max(0, min(bar_w, int(bar_w * self.bm_prob)))

        self.create_rectangle(
            pad_x, 0,
            pad_x + bar_w, 6,
            fill=PALETTE["bar_empty"],
            outline=""
        )
        if w1 > 0:
            self.create_rectangle(
                pad_x, 0,
                pad_x + w1, 6,
                fill=PALETTE["text_muted"],
                outline=""
            )


# ---------------------------------------------------------------------------
# Application Shell
# ---------------------------------------------------------------------------
class App:
    """Tkinter Desktop Application Shell for ATP Matchup Predictor."""

    def __init__(self, root: tk.Tk | None = None, init_predict: bool = True):
        if root is None:
            self.root = tk.Tk()
            self._owns_root = True
        else:
            self.root = root
            self._owns_root = False

        self.root.title("MatchPoint.intelligence | ATP Predictor")
        self.root.geometry("880x700")
        self.root.minsize(760, 580)
        self.root.configure(bg=PALETTE["bg"])

        self.font_sans = resolve_font_family(self.root, "sans")
        self.font_mono = resolve_font_family(self.root, "mono")

        # Upcoming tab variables
        self.p1_var = tk.StringVar(value=DEFAULT_P1)
        self.p2_var = tk.StringVar(value=DEFAULT_P2)
        self.surface_var = tk.StringVar(value=DEFAULT_SURFACE)
        self.series_var = tk.StringVar(value=DEFAULT_SERIES)
        self.roster_var = tk.BooleanVar(value=False)
        self.error_var = tk.StringVar(value="")
        self.last_prediction = None

        # Historical tab variables
        default_split = SPLIT_CHOICES[0]
        init_tourns = get_tournaments_for_split(default_split)
        init_tourn = init_tourns[0] if init_tourns else ""
        init_matches = get_matches_for_tournament(default_split, init_tourn) if init_tourn else []
        self.split_var = tk.StringVar(value=default_split)
        self.tourn_var = tk.StringVar(value=init_tourn)
        self.historical_matches = init_matches
        init_match_label = init_matches[0][0] if init_matches else ""
        self.match_var = tk.StringVar(value=init_match_label)
        self.historical_match_id = init_matches[0][1] if init_matches else ""
        self.historical_error_var = tk.StringVar(value="")
        self.last_historical_prediction = None

        self._predict_thread = None
        self._predict_result = None
        self._inspect_thread = None
        self._inspect_result = None
        self._init_timer = None

        self._configure_styles()
        self._build_header()
        self._build_notebook()

        if init_predict:
            self._init_timer = self.root.after(50, self.on_predict)

    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.root.option_add("*TCombobox*Listbox.background", PALETTE["card_bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", PALETTE["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", PALETTE["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.font", (self.font_sans, 10))

        style.configure("TNotebook", background=PALETTE["bg"], borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background=PALETTE["card_bg"],
            foreground=PALETTE["text_muted"],
            font=(self.font_sans, 10, "bold"),
            padding=[16, 8],
            borderwidth=1,
            bordercolor=PALETTE["border"],
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PALETTE["border"]), ("active", PALETTE["border"])],
            foreground=[("selected", PALETTE["accent"]), ("active", PALETTE["text"])],
        )

        style.configure(
            "TCombobox",
            fieldbackground=PALETTE["input_bg"],
            background=PALETTE["border"],
            foreground=PALETTE["text"],
            arrowcolor=PALETTE["text_muted"],
            darkcolor=PALETTE["border"],
            lightcolor=PALETTE["border"],
            bordercolor=PALETTE["input_border"],
            padding=6,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", PALETTE["input_bg"])],
            foreground=[("readonly", PALETTE["text"])],
            selectbackground=[("readonly", PALETTE["accent"])],
            selectforeground=[("readonly", "#ffffff")],
        )

        style.configure(
            "TCheckbutton",
            background=PALETTE["card_bg"],
            foreground=PALETTE["text"],
            font=(self.font_sans, 9),
            focuscolor=PALETTE["card_bg"],
            indicatorbackground=PALETTE["input_bg"],
            indicatorcolor=PALETTE["accent"],
        )
        style.map(
            "TCheckbutton",
            indicatorbackground=[("pressed", PALETTE["input_bg"]), ("selected", PALETTE["accent"])],
            background=[("active", PALETTE["card_bg"])],
        )

        style.configure(
            "Primary.TButton",
            background=PALETTE["accent"],
            foreground="#ffffff",
            font=(self.font_sans, 10, "bold"),
            borderwidth=0,
            focuscolor=PALETTE["accent"],
            padding=[12, 8],
        )
        style.map(
            "Primary.TButton",
            background=[("active", PALETTE["accent_hover"]), ("pressed", "#0f766e")],
            foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
        )

        style.configure(
            "Swap.TButton",
            background=PALETTE["border"],
            foreground=PALETTE["swap_btn_text"],
            font=(self.font_sans, 9, "bold"),
            borderwidth=1,
            bordercolor=PALETTE["input_border"],
            focuscolor=PALETTE["border"],
            padding=[10, 6],
        )
        style.map(
            "Swap.TButton",
            background=[("active", PALETTE["input_border"]), ("pressed", PALETTE["bg"])],
            foreground=[("active", PALETTE["swap_btn_text"])],
        )

    def _build_header(self):
        header = tk.Frame(self.root, bg=PALETTE["bg"], padx=20, pady=12)
        header.pack(fill="x")

        # Left branding
        brand_left = tk.Frame(header, bg=PALETTE["bg"])
        brand_left.pack(side="left")

        logo_frame = tk.Frame(brand_left, bg=PALETTE["accent"], width=36, height=36)
        logo_frame.pack(side="left", padx=(0, 10))
        logo_frame.pack_propagate(False)
        logo_lbl = tk.Label(logo_frame, text="MP", bg=PALETTE["accent"], fg="#ffffff", font=(self.font_sans, 12, "bold"))
        logo_lbl.pack(expand=True)

        titles_frame = tk.Frame(brand_left, bg=PALETTE["bg"])
        titles_frame.pack(side="left")

        title_lbl = tk.Label(
            titles_frame,
            text="MatchPoint.intelligence",
            bg=PALETTE["bg"],
            fg="#ffffff",
            font=(self.font_sans, 15, "bold")
        )
        title_lbl.pack(anchor="w")

        sub_lbl = tk.Label(
            titles_frame,
            text="Dual-Track ML Engine (LightGBM Champion vs LogReg Baseline) | 31 Fundamental Features",
            bg=PALETTE["bg"],
            fg=PALETTE["text_muted"],
            font=(self.font_mono, 9)
        )
        sub_lbl.pack(anchor="w")

        # Right chips
        chips_frame = tk.Frame(header, bg=PALETTE["bg"])
        chips_frame.pack(side="right")

        chip1 = tk.Frame(chips_frame, bg=PALETTE["card_bg"], highlightbackground=PALETTE["border"], highlightthickness=1, padx=8, pady=4)
        chip1.pack(side="left", padx=(0, 8))
        tk.Label(chip1, text="*", fg="#10b981", bg=PALETTE["card_bg"], font=(self.font_mono, 9, "bold")).pack(side="left", padx=(0, 4))
        tk.Label(chip1, text="Dual Models Loaded", fg=PALETTE["text"], bg=PALETTE["card_bg"], font=(self.font_mono, 8, "bold")).pack(side="left")

        chip2 = tk.Frame(chips_frame, bg=PALETTE["card_bg"], highlightbackground=PALETTE["border"], highlightthickness=1, padx=8, pady=4)
        chip2.pack(side="left")
        tk.Label(chip2, text="*", fg="#38bdf8", bg=PALETTE["card_bg"], font=(self.font_mono, 9, "bold")).pack(side="left", padx=(0, 4))
        tk.Label(chip2, text="H2H History 2000-2026 Ready", fg=PALETTE["text"], bg=PALETTE["card_bg"], font=(self.font_mono, 8, "bold")).pack(side="left")

        # Divider
        divider = tk.Frame(self.root, bg=PALETTE["border"], height=1)
        divider.pack(fill="x", padx=20, pady=(0, 10))

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.tab_upcoming = tk.Frame(self.notebook, bg=PALETTE["bg"], padx=4, pady=4)
        self.tab_historical = tk.Frame(self.notebook, bg=PALETTE["bg"], padx=4, pady=4)
        self.notebook.add(self.tab_upcoming, text="Upcoming Predictor")
        self.notebook.add(self.tab_historical, text="Historical Backtracker")
        self.notebook.pack(fill="both", expand=True, padx=20, pady=(0, 16))

        self._build_upcoming_tab()
        self._build_historical_tab()

    def _build_upcoming_tab(self):
        # 1. Controls Card
        card_controls = tk.Frame(
            self.tab_upcoming,
            bg=PALETTE["card_bg"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=18,
            pady=16
        )
        card_controls.pack(fill="x", pady=(0, 14))

        # Row 0: Roster Toggle
        self.roster_cb = ttk.Checkbutton(
            card_controls,
            text="Unlock All-Time Historical Roster (2000-2026)",
            variable=self.roster_var,
            command=self.on_roster_toggle,
            style="TCheckbutton"
        )
        self.roster_cb.pack(anchor="w", pady=(0, 12))

        # Row 1: Player Selection Frame
        p_row = tk.Frame(card_controls, bg=PALETTE["card_bg"])
        p_row.pack(fill="x", pady=(0, 12))
        p_row.columnconfigure(0, weight=5)
        p_row.columnconfigure(1, weight=0)
        p_row.columnconfigure(2, weight=5)

        p1_box = tk.Frame(p_row, bg=PALETTE["card_bg"])
        p1_box.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(p1_box, text="PLAYER 1", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        self.p1_combo = ttk.Combobox(p1_box, textvariable=self.p1_var, values=active_players, font=(self.font_sans, 10))
        self.p1_combo.pack(fill="x")

        swap_box = tk.Frame(p_row, bg=PALETTE["card_bg"])
        swap_box.grid(row=0, column=1, sticky="s", padx=6, pady=(0, 1))
        self.swap_btn = ttk.Button(swap_box, text="<-> Swap", style="Swap.TButton", command=self.on_swap)
        self.swap_btn.pack()

        p2_box = tk.Frame(p_row, bg=PALETTE["card_bg"])
        p2_box.grid(row=0, column=2, sticky="ew", padx=(6, 0))
        tk.Label(p2_box, text="PLAYER 2", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        self.p2_combo = ttk.Combobox(p2_box, textvariable=self.p2_var, values=active_players, font=(self.font_sans, 10))
        self.p2_combo.pack(fill="x")

        # Row 2: Match Conditions and Predict
        c_row = tk.Frame(card_controls, bg=PALETTE["card_bg"])
        c_row.pack(fill="x")
        c_row.columnconfigure(0, weight=4)
        c_row.columnconfigure(1, weight=4)
        c_row.columnconfigure(2, weight=4)

        surf_box = tk.Frame(c_row, bg=PALETTE["card_bg"])
        surf_box.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(surf_box, text="COURT SURFACE", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        self.surface_combo = ttk.Combobox(surf_box, textvariable=self.surface_var, values=SURFACE_CHOICES, state="readonly", font=(self.font_sans, 10))
        self.surface_combo.pack(fill="x")

        series_box = tk.Frame(c_row, bg=PALETTE["card_bg"])
        series_box.grid(row=0, column=1, sticky="ew", padx=(6, 6))
        tk.Label(series_box, text="TOURNAMENT SERIES", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        self.series_combo = ttk.Combobox(series_box, textvariable=self.series_var, values=SERIES_CHOICES, state="readonly", font=(self.font_sans, 10))
        self.series_combo.pack(fill="x")

        predict_box = tk.Frame(c_row, bg=PALETTE["card_bg"])
        predict_box.grid(row=0, column=2, sticky="se", padx=(6, 0), pady=(0, 1))
        self.predict_btn = ttk.Button(predict_box, text="Predict Matchup", style="Primary.TButton", command=self.on_predict)
        self.predict_btn.pack(fill="x")

        # 2. Results Card
        self.card_results = tk.Frame(
            self.tab_upcoming,
            bg=PALETTE["card_bg"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=18,
            pady=16
        )
        self.card_results.pack(fill="both", expand=True)

        card_title = tk.Label(
            self.card_results,
            text="MODEL DIAGNOSTICS & WIN PROBABILITIES",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_dim"],
            font=(self.font_mono, 9, "bold")
        )
        card_title.pack(anchor="w", pady=(0, 10))

        # Error banner
        self.error_frame = tk.Frame(
            self.card_results,
            bg=PALETTE["error_bg"],
            highlightbackground=PALETTE["error_text"],
            highlightthickness=1,
            padx=14,
            pady=10
        )
        self.error_label = tk.Label(
            self.error_frame,
            textvariable=self.error_var,
            bg=PALETTE["error_bg"],
            fg="#fb7185",
            font=(self.font_sans, 10, "bold"),
            wraplength=600,
            justify="left"
        )
        self.error_label.pack(anchor="w")

        # Matchup header
        self.header_frame = tk.Frame(self.card_results, bg=PALETTE["card_bg"])
        hdr_left = tk.Frame(self.header_frame, bg=PALETTE["card_bg"])
        hdr_left.pack(side="left")

        self.matchup_label = tk.Label(
            hdr_left,
            text="",
            bg=PALETTE["card_bg"],
            fg="#ffffff",
            font=(self.font_sans, 13, "bold")
        )
        self.matchup_label.pack(anchor="w")

        self.subtitle_label = tk.Label(
            hdr_left,
            text="",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_muted"],
            font=(self.font_mono, 9)
        )
        self.subtitle_label.pack(anchor="w")

        self.badges_frame = tk.Frame(self.header_frame, bg=PALETTE["card_bg"])
        self.badges_frame.pack(side="right")

        self.surface_badge = tk.Label(
            self.badges_frame,
            text="",
            bg=PALETTE["border"],
            fg=PALETTE["accent"],
            font=(self.font_mono, 9, "bold"),
            padx=8,
            pady=2
        )
        self.surface_badge.pack(side="left", padx=(0, 6))

        self.series_badge = tk.Label(
            self.badges_frame,
            text="",
            bg=PALETTE["border"],
            fg=PALETTE["text"],
            font=(self.font_mono, 9),
            padx=8,
            pady=2
        )
        self.series_badge.pack(side="left", padx=(0, 6))

        self.symmetry_badge = tk.Label(
            self.badges_frame,
            text="",
            bg=PALETTE["border"],
            fg="#10b981",
            font=(self.font_mono, 9, "bold"),
            padx=8,
            pady=2
        )
        self.symmetry_badge.pack(side="left")

        # Surface accent strip
        self.surface_accent_strip = tk.Frame(self.card_results, height=3, bg=PALETTE["accent"])

        # Divergence warning alert frame
        self.divergence_frame = tk.Frame(
            self.card_results,
            bg="#241b0a",
            highlightbackground="#f59e0b",
            highlightthickness=1,
            padx=12,
            pady=8
        )
        self.divergence_icon = tk.Label(
            self.divergence_frame,
            text="[!]",
            bg="#241b0a",
            fg="#f59e0b",
            font=(self.font_mono, 9, "bold")
        )
        self.divergence_icon.pack(side="left", padx=(0, 8))
        self.divergence_label = tk.Label(
            self.divergence_frame,
            text="",
            bg="#241b0a",
            fg="#fde68a",
            font=(self.font_sans, 9),
            wraplength=620,
            justify="left"
        )
        self.divergence_label.pack(side="left", fill="x", expand=True)
        self.is_divergent_visible = False

        # Probability Canvas
        self.canvas_bars = ProbabilityBarsCanvas(
            self.card_results,
            font_sans=self.font_sans,
            font_mono=self.font_mono
        )
        self.prob_display = self.canvas_bars

        # 5 Key Delta Summary Tiles Frame
        self.stats_frame = tk.Frame(
            self.card_results,
            bg="#0b0f19",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=10,
            pady=10
        )
        for col in range(5):
            self.stats_frame.columnconfigure(col, weight=1)

        self.stats_tiles = {}
        for i, key in enumerate(["rank", "swr", "form", "fatigue", "h2h"]):
            tile = tk.Frame(self.stats_frame, bg="#0b0f19")
            tile.grid(row=0, column=i, sticky="nsew", padx=4)
            lbl_title = tk.Label(
                tile,
                text="",
                bg="#0b0f19",
                fg=PALETTE["text_dim"],
                font=(self.font_mono, 8, "bold")
            )
            lbl_title.pack()
            lbl_val = tk.Label(
                tile,
                text="",
                bg="#0b0f19",
                fg=PALETTE["text"],
                font=(self.font_mono, 12, "bold")
            )
            lbl_val.pack(pady=(2, 0))
            self.stats_tiles[key] = {
                "frame": tile,
                "title": lbl_title,
                "val": lbl_val,
            }

        # Placeholder
        self.placeholder_frame = tk.Frame(self.card_results, bg=PALETTE["card_bg"], pady=30)
        self.placeholder_label = tk.Label(
            self.placeholder_frame,
            text="Ready. Select players and court conditions, then click 'Predict Matchup'.",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_dim"],
            font=(self.font_sans, 10, "italic")
        )
        self.placeholder_label.pack()
        self.placeholder_frame.pack(fill="both", expand=True)

    def _build_historical_tab(self):
        # 1. Controls Card
        card_hist_controls = tk.Frame(
            self.tab_historical,
            bg=PALETTE["card_bg"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=18,
            pady=16
        )
        card_hist_controls.pack(fill="x", pady=(0, 14))

        # Row 0: Split and Tournament selection
        row0 = tk.Frame(card_hist_controls, bg=PALETTE["card_bg"])
        row0.pack(fill="x", pady=(0, 12))
        row0.columnconfigure(0, weight=1)
        row0.columnconfigure(1, weight=1)

        split_box = tk.Frame(row0, bg=PALETTE["card_bg"])
        split_box.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(split_box, text="DATASET SPLIT", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        self.split_combo = ttk.Combobox(split_box, textvariable=self.split_var, values=SPLIT_CHOICES, state="readonly", font=(self.font_sans, 10))
        self.split_combo.pack(fill="x")

        tourn_box = tk.Frame(row0, bg=PALETTE["card_bg"])
        tourn_box.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(tourn_box, text="TOURNAMENT", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        init_tourns = get_tournaments_for_split(self.split_var.get())
        self.tourn_combo = ttk.Combobox(tourn_box, textvariable=self.tourn_var, values=init_tourns, state="readonly", font=(self.font_sans, 10))
        self.tourn_combo.pack(fill="x")

        # Row 1: Match selection and Inspect button
        row1 = tk.Frame(card_hist_controls, bg=PALETTE["card_bg"])
        row1.pack(fill="x")
        row1.columnconfigure(0, weight=4)
        row1.columnconfigure(1, weight=1)

        match_box = tk.Frame(row1, bg=PALETTE["card_bg"])
        match_box.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Label(match_box, text="MATCH FIXTURE & OUTCOME", bg=PALETTE["card_bg"], fg=PALETTE["text_muted"], font=(self.font_sans, 8, "bold")).pack(anchor="w", pady=(0, 4))
        match_labels = [m[0] for m in self.historical_matches]
        self.match_combo = ttk.Combobox(match_box, textvariable=self.match_var, values=match_labels, state="readonly", font=(self.font_sans, 10))
        self.match_combo.pack(fill="x")

        btn_box = tk.Frame(row1, bg=PALETTE["card_bg"])
        btn_box.grid(row=0, column=1, sticky="se", padx=(6, 0), pady=(0, 1))
        self.inspect_btn = ttk.Button(btn_box, text="Inspect & Backtrack", style="Primary.TButton", command=self.on_inspect)
        self.inspect_btn.pack(fill="x")
        self.hist_inspect_btn = self.inspect_btn

        # Bind cascading changes
        self.split_combo.bind("<<ComboboxSelected>>", self.on_historical_split_change)
        self.tourn_combo.bind("<<ComboboxSelected>>", self.on_historical_tournament_change)

        # 2. Results Card
        self.card_hist_results = tk.Frame(
            self.tab_historical,
            bg=PALETTE["card_bg"],
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=18,
            pady=16
        )
        self.card_hist_results.pack(fill="both", expand=True)

        card_title = tk.Label(
            self.card_hist_results,
            text="HISTORICAL FIXTURE DIAGNOSTICS & VERIFICATION",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_dim"],
            font=(self.font_mono, 9, "bold")
        )
        card_title.pack(anchor="w", pady=(0, 10))

        # Error banner
        self.hist_error_frame = tk.Frame(
            self.card_hist_results,
            bg=PALETTE["error_bg"],
            highlightbackground=PALETTE["error_text"],
            highlightthickness=1,
            padx=14,
            pady=10
        )
        self.hist_error_label = tk.Label(
            self.hist_error_frame,
            textvariable=self.historical_error_var,
            bg=PALETTE["error_bg"],
            fg="#fb7185",
            font=(self.font_sans, 10, "bold"),
            wraplength=600,
            justify="left"
        )
        self.hist_error_label.pack(anchor="w")

        # Surface accent strip
        self.hist_surface_accent_strip = tk.Frame(self.card_hist_results, height=3, bg=PALETTE["accent"])

        # Header frame
        self.hist_header_frame = tk.Frame(self.card_hist_results, bg=PALETTE["card_bg"])
        hdr_left = tk.Frame(self.hist_header_frame, bg=PALETTE["card_bg"])
        hdr_left.pack(side="left")

        self.hist_matchup_label = tk.Label(
            hdr_left,
            text="",
            bg=PALETTE["card_bg"],
            fg="#ffffff",
            font=(self.font_sans, 13, "bold")
        )
        self.hist_matchup_label.pack(anchor="w")

        self.hist_subtitle_label = tk.Label(
            hdr_left,
            text="",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_muted"],
            font=(self.font_mono, 9)
        )
        self.hist_subtitle_label.pack(anchor="w")

        self.hist_badges_frame = tk.Frame(self.hist_header_frame, bg=PALETTE["card_bg"])
        self.hist_badges_frame.pack(side="right")

        self.hist_surface_badge = tk.Label(
            self.hist_badges_frame,
            text="",
            bg=PALETTE["border"],
            fg=PALETTE["accent"],
            font=(self.font_mono, 9, "bold"),
            padx=8,
            pady=2
        )
        self.hist_surface_badge.pack(side="left", padx=(0, 6))

        self.hist_series_badge = tk.Label(
            self.hist_badges_frame,
            text="",
            bg=PALETTE["border"],
            fg=PALETTE["text"],
            font=(self.font_mono, 9),
            padx=8,
            pady=2
        )
        self.hist_series_badge.pack(side="left")

        # Winner badge frame
        self.hist_winner_badge_frame = tk.Frame(
            self.card_hist_results,
            bg=PALETTE["correct_bg"],
            highlightbackground=PALETTE["correct_border"],
            highlightthickness=1,
            padx=14,
            pady=10
        )
        self.hist_winner_badge_left = tk.Frame(self.hist_winner_badge_frame, bg=PALETTE["correct_bg"])
        self.hist_winner_badge_left.pack(side="left", fill="x", expand=True)

        self.hist_winner_icon = tk.Label(
            self.hist_winner_badge_left,
            text="[OK]",
            bg=PALETTE["correct_bg"],
            fg=PALETTE["correct_text"],
            font=(self.font_mono, 9, "bold")
        )
        self.hist_winner_icon.pack(side="left", padx=(0, 8))

        self.hist_winner_text = tk.Label(
            self.hist_winner_badge_left,
            text="",
            bg=PALETTE["correct_bg"],
            fg="#ffffff",
            font=(self.font_sans, 10, "bold")
        )
        self.hist_winner_text.pack(side="left")

        self.hist_score_text = tk.Label(
            self.hist_winner_badge_left,
            text="",
            bg=PALETTE["correct_bg"],
            fg=PALETTE["text_muted"],
            font=(self.font_mono, 9)
        )
        self.hist_score_text.pack(side="left", padx=(10, 0))

        self.hist_winner_badge_right = tk.Frame(self.hist_winner_badge_frame, bg=PALETTE["correct_bg"])
        self.hist_winner_badge_right.pack(side="right")

        self.hist_status_pill = tk.Label(
            self.hist_winner_badge_right,
            text="",
            bg=PALETTE["correct_border"],
            fg="#0b0f19",
            font=(self.font_mono, 9, "bold"),
            padx=8,
            pady=2
        )
        self.hist_status_pill.pack()

        # Divergence warning banner
        self.hist_divergence_frame = tk.Frame(
            self.card_hist_results,
            bg="#241b0a",
            highlightbackground="#f59e0b",
            highlightthickness=1,
            padx=12,
            pady=8
        )
        self.hist_divergence_icon = tk.Label(
            self.hist_divergence_frame,
            text="[!]",
            bg="#241b0a",
            fg="#f59e0b",
            font=(self.font_mono, 9, "bold")
        )
        self.hist_divergence_icon.pack(side="left", padx=(0, 8))
        self.hist_divergence_label = tk.Label(
            self.hist_divergence_frame,
            text="",
            bg="#241b0a",
            fg="#fde68a",
            font=(self.font_sans, 9),
            wraplength=620,
            justify="left"
        )
        self.hist_divergence_label.pack(side="left", fill="x", expand=True)
        self.hist_is_divergent_visible = False

        # Bookmaker implied odds section
        self.hist_bm_frame = tk.Frame(self.card_hist_results, bg=PALETTE["card_bg"])
        bm_hdr = tk.Frame(self.hist_bm_frame, bg=PALETTE["card_bg"])
        bm_hdr.pack(fill="x", pady=(0, 4))
        tk.Label(
            bm_hdr,
            text="Market Implied (Bookmaker Odds)",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_muted"],
            font=(self.font_mono, 9)
        ).pack(side="left")
        self.hist_bm_val_label = tk.Label(
            bm_hdr,
            text="",
            bg=PALETTE["card_bg"],
            fg="#cbd5e1",
            font=(self.font_mono, 9, "bold")
        )
        self.hist_bm_val_label.pack(side="right")
        self.hist_bm_canvas = BookmakerBarCanvas(self.hist_bm_frame)
        self.hist_bm_canvas.pack(fill="x")

        # Dual model probability bars
        self.hist_canvas_bars = ProbabilityBarsCanvas(
            self.card_hist_results,
            font_sans=self.font_sans,
            font_mono=self.font_mono
        )

        # 5 Key Delta Summary Tiles Frame
        self.hist_stats_frame = tk.Frame(
            self.card_hist_results,
            bg="#0b0f19",
            highlightbackground=PALETTE["border"],
            highlightthickness=1,
            padx=10,
            pady=10
        )
        for col in range(5):
            self.hist_stats_frame.columnconfigure(col, weight=1)

        self.hist_stats_tiles = {}
        for i, key in enumerate(["rank", "swr", "form", "fatigue", "h2h"]):
            tile = tk.Frame(self.hist_stats_frame, bg="#0b0f19")
            tile.grid(row=0, column=i, sticky="nsew", padx=4)
            lbl_title = tk.Label(
                tile,
                text="",
                bg="#0b0f19",
                fg=PALETTE["text_dim"],
                font=(self.font_mono, 8, "bold")
            )
            lbl_title.pack()
            lbl_val = tk.Label(
                tile,
                text="",
                bg="#0b0f19",
                fg=PALETTE["text"],
                font=(self.font_mono, 12, "bold")
            )
            lbl_val.pack(pady=(2, 0))
            self.hist_stats_tiles[key] = {
                "frame": tile,
                "title": lbl_title,
                "val": lbl_val,
            }

        # Placeholder
        self.hist_placeholder_frame = tk.Frame(self.card_hist_results, bg=PALETTE["card_bg"], pady=30)
        self.hist_placeholder_label = tk.Label(
            self.hist_placeholder_frame,
            text="Ready. Select split, tournament, and matchup, then click 'Inspect & Backtrack'.",
            bg=PALETTE["card_bg"],
            fg=PALETTE["text_dim"],
            font=(self.font_sans, 10, "italic")
        )
        self.hist_placeholder_label.pack()
        self.hist_placeholder_frame.pack(fill="both", expand=True)

    # -----------------------------------------------------------------------
    # Event Handlers
    # -----------------------------------------------------------------------
    def on_swap(self):
        cur_p1 = self.p1_var.get()
        cur_p2 = self.p2_var.get()
        new_p1, new_p2 = resolve_swap(cur_p1, cur_p2)
        self.p1_var.set(new_p1)
        self.p2_var.set(new_p2)

    def on_roster_toggle(self):
        unlock_all = bool(self.roster_var.get())
        cur_p1 = self.p1_var.get()
        cur_p2 = self.p2_var.get()
        pool, new_p1, new_p2 = resolve_roster_toggle(unlock_all, cur_p1, cur_p2)
        self.p1_combo["values"] = pool
        self.p2_combo["values"] = pool
        self.p1_var.set(new_p1)
        self.p2_var.set(new_p2)

    def on_predict(self, *args, async_run: bool = True, **kwargs) -> threading.Thread | None:
        if getattr(self, "_init_timer", None) is not None:
            try:
                self.root.after_cancel(self._init_timer)
            except Exception:
                pass
            self._init_timer = None

        if args and isinstance(args[0], bool):
            async_run = args[0]

        p1 = self.p1_var.get().strip()
        p2 = self.p2_var.get().strip()
        surface = self.surface_var.get().strip()
        series = self.series_var.get().strip()

        self._predict_result = None
        self.predict_btn.config(state="disabled")
        self.hist_inspect_btn.config(state="disabled")

        if not async_run:
            res = None
            exc = None
            try:
                res = atp_service.predict(p1, p2, surface, series)
            except Exception as e:
                exc = e
            self._on_prediction_done(res, exc)
            return None

        def worker():
            res = None
            exc = None
            try:
                res = atp_service.predict(p1, p2, surface, series)
            except Exception as e:
                exc = e
            self._predict_result = (res, exc)
            try:
                self.root.after(0, lambda: self._on_prediction_done(res, exc))
            except RuntimeError:
                pass

        thread = threading.Thread(target=worker, daemon=True)
        self._predict_thread = thread
        thread.start()
        return thread

    def _on_prediction_done(self, res: dict | None = None, exc: Exception | None = None):
        self._predict_result = None
        try:
            if exc is not None:
                self.show_error(str(exc))
            elif res is not None:
                self.last_prediction = res
                err = format_error_text(res)
                if err:
                    self.show_error(err)
                else:
                    self.show_prediction(res)
        finally:
            self.predict_btn.config(state="normal")
            self.hist_inspect_btn.config(state="normal")

    def reset_card(self):
        self.error_var.set("")
        self.is_divergent_visible = False
        self.surface_accent_strip.pack_forget()
        self.header_frame.pack_forget()
        self.divergence_frame.pack_forget()
        self.canvas_bars.pack_forget()
        self.stats_frame.pack_forget()
        self.error_frame.pack_forget()
        self.card_results.config(highlightbackground=PALETTE["border"])
        self.placeholder_frame.pack(fill="both", expand=True)

    def show_error(self, message: str):
        self.error_var.set(f"[!] {message}")
        self.is_divergent_visible = False
        self.placeholder_frame.pack_forget()
        self.surface_accent_strip.pack_forget()
        self.header_frame.pack_forget()
        self.divergence_frame.pack_forget()
        self.canvas_bars.pack_forget()
        self.stats_frame.pack_forget()
        self.card_results.config(highlightbackground=PALETTE["border"])
        self.error_frame.pack(fill="x", pady=(0, 12))

    def show_prediction(self, res: dict):
        self.error_var.set("")
        self.error_frame.pack_forget()
        self.placeholder_frame.pack_forget()

        p1 = res.get("p1", "")
        p2 = res.get("p2", "")
        surface = res.get("surface", "")
        series = res.get("series", "")
        prob_lgb = float(res.get("prob_lgb", 0.5))
        prob_lr = float(res.get("prob_lr", 0.5))
        key_stats = res.get("key_stats", {})
        sym_score = float(res.get("sym_score", 0.0))
        raw_asym = float(res.get("raw_asym", 0.0))
        div_delta = float(res.get("div_delta", abs(prob_lgb - prob_lr)))
        is_divergent = bool(res.get("is_divergent", div_delta > 0.15))

        self.matchup_label.config(text=f"{p1}  vs  {p2}")
        self.subtitle_label.config(text=f"{series} | {surface} Court")
        surf_color = resolve_surface_color(surface)
        self.surface_badge.config(text=surface, fg=surf_color)
        self.series_badge.config(text=series)
        self.symmetry_badge.config(text=format_symmetry_text(raw_asym, sym_score))
        self.surface_accent_strip.config(bg=surf_color)
        self.card_results.config(highlightbackground=surf_color)

        self.surface_accent_strip.pack(fill="x", pady=(0, 10))
        self.header_frame.pack(fill="x", pady=(0, 10))

        is_div, div_msg = resolve_divergence_status(is_divergent, div_delta)
        self.is_divergent_visible = is_div
        if is_div:
            self.divergence_label.config(text=div_msg)
            self.divergence_frame.pack(fill="x", pady=(0, 10))
        else:
            self.divergence_frame.pack_forget()

        self.canvas_bars.set_data(p1, p2, prob_lgb, prob_lr)
        self.canvas_bars.pack(fill="x", pady=(0, 8))

        formatted_stats = format_key_stats(key_stats)
        for item in formatted_stats:
            k = item["id"]
            if k in self.stats_tiles:
                self.stats_tiles[k]["title"].config(text=item["label"].upper())
                self.stats_tiles[k]["val"].config(text=item["val"], fg=item["color"])
        self.stats_frame.pack(fill="x", pady=(8, 0))

    def on_historical_split_change(self, event=None):
        new_split = self.split_var.get()
        tourns, first_tourn, matches, first_match_id = resolve_split_change(new_split)
        self.tourn_combo["values"] = tourns
        self.tourn_var.set(first_tourn)
        self.historical_matches = matches
        match_labels = [m[0] for m in matches]
        self.match_combo["values"] = match_labels
        self.match_var.set(match_labels[0] if match_labels else "")
        self.historical_match_id = first_match_id

    def on_historical_tournament_change(self, event=None):
        cur_split = self.split_var.get()
        tourn_name = self.tourn_var.get()
        matches, first_match_id = resolve_tournament_change(cur_split, tourn_name)
        self.historical_matches = matches
        match_labels = [m[0] for m in matches]
        self.match_combo["values"] = match_labels
        self.match_var.set(match_labels[0] if match_labels else "")
        self.historical_match_id = first_match_id

    def on_inspect(self, *args, async_run: bool = True, **kwargs) -> threading.Thread | None:
        if args and isinstance(args[0], bool):
            async_run = args[0]

        cur_split = self.split_var.get().strip()
        tourn_name = self.tourn_var.get().strip()
        match_label = self.match_var.get().strip()

        match_id = ""
        if match_label and self.historical_matches:
            for label, mid in self.historical_matches:
                if label == match_label:
                    match_id = mid
                    break
            if not match_id and self.match_combo.current() >= 0:
                idx = self.match_combo.current()
                if idx < len(self.historical_matches):
                    match_id = self.historical_matches[idx][1]

        if not match_id or not tourn_name or not cur_split:
            self.show_historical_error("Select a valid tournament and matchup to inspect.")
            return None

        self._inspect_result = None
        self.predict_btn.config(state="disabled")
        self.hist_inspect_btn.config(state="disabled")

        if not async_run:
            res = None
            exc = None
            try:
                res = atp_service.inspect(cur_split, tourn_name, match_id)
            except Exception as e:
                exc = e
            self._on_inspect_done(res, exc)
            return None

        def worker():
            res = None
            exc = None
            try:
                res = atp_service.inspect(cur_split, tourn_name, match_id)
            except Exception as e:
                exc = e
            self._inspect_result = (res, exc)
            try:
                self.root.after(0, lambda: self._on_inspect_done(res, exc))
            except RuntimeError:
                pass

        thread = threading.Thread(target=worker, daemon=True)
        self._inspect_thread = thread
        thread.start()
        return thread

    def _on_inspect_done(self, res: dict | None = None, exc: Exception | None = None):
        self._inspect_result = None
        try:
            if exc is not None:
                self.show_historical_error(str(exc))
            elif res is not None:
                self.last_historical_prediction = res
                err = format_error_text(res)
                if err:
                    self.show_historical_error(err)
                else:
                    self.show_historical_prediction(res)
        finally:
            self.predict_btn.config(state="normal")
            self.hist_inspect_btn.config(state="normal")

    _on_historical_done = _on_inspect_done

    def reset_historical_card(self):
        self.historical_error_var.set("")
        self.hist_is_divergent_visible = False
        self.hist_surface_accent_strip.pack_forget()
        self.hist_header_frame.pack_forget()
        self.hist_winner_badge_frame.pack_forget()
        self.hist_bm_frame.pack_forget()
        self.hist_divergence_frame.pack_forget()
        self.hist_canvas_bars.pack_forget()
        self.hist_stats_frame.pack_forget()
        self.hist_error_frame.pack_forget()
        self.card_hist_results.config(highlightbackground=PALETTE["border"])
        self.hist_placeholder_frame.pack(fill="both", expand=True)

    def show_historical_error(self, message: str):
        self.historical_error_var.set(f"[!] {message}")
        self.hist_is_divergent_visible = False
        self.hist_placeholder_frame.pack_forget()
        self.hist_surface_accent_strip.pack_forget()
        self.hist_header_frame.pack_forget()
        self.hist_winner_badge_frame.pack_forget()
        self.hist_bm_frame.pack_forget()
        self.hist_divergence_frame.pack_forget()
        self.hist_canvas_bars.pack_forget()
        self.hist_stats_frame.pack_forget()
        self.card_hist_results.config(highlightbackground=PALETTE["border"])
        self.hist_error_frame.pack(fill="x", pady=(0, 12))

    def show_historical_prediction(self, res: dict):
        self.historical_error_var.set("")
        self.hist_error_frame.pack_forget()
        self.hist_placeholder_frame.pack_forget()

        p1 = res.get("p1", "")
        p2 = res.get("p2", "")
        surface = res.get("surface", "Hard")
        series = res.get("series", "Grand Slam")
        prob_lgb = float(res.get("prob_lgb", 0.5))
        prob_lr = float(res.get("prob_lr", 0.5))
        actual_winner = res.get("actual_winner", "")
        score_str = res.get("score_str", "")
        date_str = res.get("date_str", "")
        round_name = res.get("round_name", "")
        tournament = res.get("tournament", "")
        bm_prob = res.get("bm_prob", None)
        key_stats = res.get("key_stats", {})
        div_delta = float(res.get("div_delta", abs(prob_lgb - prob_lr)))
        is_divergent = bool(res.get("is_divergent", div_delta > 0.15))

        # Accent strip & card highlight
        surf_color = resolve_surface_color(surface)
        self.hist_surface_accent_strip.config(bg=surf_color)
        self.card_hist_results.config(highlightbackground=surf_color)
        self.hist_surface_accent_strip.pack(fill="x", pady=(0, 10))

        # Matchup header
        self.hist_matchup_label.config(text=f"{p1}  vs  {p2}")
        round_part = f" ({round_name})" if round_name else ""
        sub_txt = f"{date_str} | {tournament}{round_part} | {surface}" if tournament and date_str else f"{series} | {surface} Court"
        self.hist_subtitle_label.config(text=sub_txt)
        self.hist_surface_badge.config(text=surface, fg=surf_color)
        self.hist_series_badge.config(text=series)
        self.hist_header_frame.pack(fill="x", pady=(0, 10))

        # Winner badge
        outcome = evaluate_historical_outcome(p1, actual_winner, prob_lgb)
        self.hist_winner_badge_frame.config(
            bg=outcome["badge_bg"],
            highlightbackground=outcome["badge_border"],
        )
        self.hist_winner_badge_left.config(bg=outcome["badge_bg"])
        self.hist_winner_badge_right.config(bg=outcome["badge_bg"])
        self.hist_winner_icon.config(
            text=outcome["icon"],
            bg=outcome["badge_bg"],
            fg=outcome["badge_text_color"],
        )
        self.hist_winner_text.config(
            text=f"Actual Winner: {actual_winner}",
            bg=outcome["badge_bg"],
        )
        if score_str:
            self.hist_score_text.config(
                text=f"Score: {score_str}",
                bg=outcome["badge_bg"],
            )
            self.hist_score_text.pack(side="left", padx=(10, 0))
        else:
            self.hist_score_text.pack_forget()

        self.hist_status_pill.config(
            text=f"{outcome['icon']} {outcome['badge_text']}",
            bg=outcome["badge_border"],
            fg="#0b0f19",
        )
        self.hist_winner_badge_frame.pack(fill="x", pady=(0, 10))

        # Divergence warning banner
        is_div, div_msg = resolve_divergence_status(is_divergent, div_delta)
        self.hist_is_divergent_visible = is_div
        if is_div:
            self.hist_divergence_label.config(text=div_msg)
            self.hist_divergence_frame.pack(fill="x", pady=(0, 10))
        else:
            self.hist_divergence_frame.pack_forget()

        # Bookmaker implied odds
        bm_data = format_bm_odds(bm_prob)
        if bm_data is not None:
            self.hist_bm_val_label.config(text=f"{p1}: {bm_data['bm_p1_pct']}   |   {p2}: {bm_data['bm_p2_pct']}")
            self.hist_bm_canvas.set_prob(bm_prob)
            self.hist_bm_frame.pack(fill="x", pady=(0, 10))
        else:
            self.hist_bm_frame.pack_forget()

        # Dual model probability bars
        self.hist_canvas_bars.set_data(p1, p2, prob_lgb, prob_lr)
        self.hist_canvas_bars.pack(fill="x", pady=(0, 8))

        # Key stats tiles
        formatted_stats = format_key_stats(key_stats)
        for item in formatted_stats:
            k = item["id"]
            if k in self.hist_stats_tiles:
                self.hist_stats_tiles[k]["title"].config(text=item["label"].upper())
                self.hist_stats_tiles[k]["val"].config(text=item["val"], fg=item["color"])
        self.hist_stats_frame.pack(fill="x", pady=(8, 0))

    def wait_for_prediction(self, timeout: float = 5.0):
        """Waits for prediction thread or startup prediction to finish and processes Tk events."""
        start = time.time()
        while getattr(self, "_init_timer", None) is not None and getattr(self, "_predict_thread", None) is None and (time.time() - start) < timeout:
            self.root.update()
            time.sleep(0.01)

        t = getattr(self, "_predict_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

        if getattr(self, "_predict_result", None) is not None:
            r, e = self._predict_result
            self._predict_result = None
            self._on_prediction_done(r, e)
        self.root.update()

    def wait_for_inspect(self, timeout: float = 5.0):
        """Waits for historical inspect thread to finish and processes Tk events."""
        t = getattr(self, "_inspect_thread", None)
        if t is not None and t.is_alive():
            t.join(timeout=timeout)

        if getattr(self, "_inspect_result", None) is not None:
            r, e = self._inspect_result
            self._inspect_result = None
            self._on_inspect_done(r, e)
        self.root.update()

    def mainloop(self):
        self.root.mainloop()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
