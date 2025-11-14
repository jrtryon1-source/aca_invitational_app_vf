import streamlit as st
import pandas as pd
import random
import ast
from pathlib import Path
from datetime import time
from io import StringIO
import re

# ==========================
# Page config
# ==========================
st.set_page_config(page_title="Tournament Hub", layout="wide")

# ==========================
# Data folder / files
# ==========================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

STUDENTS_FILE = Path(__file__).parent / "student_list" / "students.csv"
ROOMS_FILE = Path(__file__).parent / "student_list" / "rooms.csv"

# ==========================
# Load / Save helpers
# ==========================
def load_students():
    if "students" not in st.session_state:
        if STUDENTS_FILE.exists():
            df = pd.read_csv(STUDENTS_FILE)
        else:
            df = pd.DataFrame(columns=[
                "First Name", "Last Name", "Full Name", "School",
                "Wins", "Losses", "Average Speaker Points"
            ])

        # Ensure required numeric columns exist and are zeroed
        for col in ["Wins", "Losses", "Average Speaker Points"]:
            if col not in df.columns:
                df[col] = 0

        df["Wins"] = pd.to_numeric(df["Wins"], errors="coerce").fillna(0).astype(int)
        df["Losses"] = pd.to_numeric(df["Losses"], errors="coerce").fillna(0).astype(int)
        df["Average Speaker Points"] = pd.to_numeric(
            df["Average Speaker Points"], errors="coerce"
        ).fillna(0.0)

        st.session_state["students"] = df

    return st.session_state["students"]


def save_students():
    if "students" in st.session_state:
        st.session_state["students"].to_csv(STUDENTS_FILE, index=False)


def load_judges():
    if "judges" not in st.session_state:
        st.session_state["judges"] = pd.DataFrame(columns=[
            "First Name", "Last Name", "Full Name", "Cannot Judge", "Judging Rounds"
        ])
    return st.session_state["judges"]


def load_rooms():
    if "rooms" not in st.session_state:
        # "Rounds Available" can be "All" or list-like (['1','2',...])
        st.session_state["rooms"] = pd.DataFrame(columns=[
            "Room Number", "Rounds Available"
        ])
    return st.session_state["rooms"]


def save_rooms():
    if "rooms" in st.session_state:
        st.session_state["rooms"].to_csv(ROOMS_FILE, index=False)


def load_schedule():
    if "schedule" not in st.session_state:
        st.session_state["schedule"] = pd.DataFrame({
            "Round": ["Round 1 🏛️", "Round 2 🏛️", "Round 3 🏛️", "Round 4 🏛️", "Semifinals 🏆", "Final 🏆"],
            "Time": ["09:00 AM", "10:30 AM", "12:00 PM", "01:30 PM", "03:00 PM", "04:30 PM"]
        })
    return st.session_state["schedule"]


def load_rounds():
    if "rounds" not in st.session_state:
        st.session_state["rounds"] = pd.DataFrame(columns=[
            "Round", "Student", "Opponent", "Side", "Win", "Speaker Points", "Room", "Judge"
        ])
    return st.session_state["rounds"]

def render_printable_postings(round_df: pd.DataFrame, round_number: int):
    """
    Shows a clean AFF | NEG | ROOM table for the round, with:
      • Print button (opens browser print dialog)
      • Download CSV button
    Expects columns: ["Round","AFF","NEG","ROOM"] in round_df.
    """
    if round_df.empty:
        st.info("No pairings to print for this round yet.")
        return

    # Keep only the three columns (fallback to blank if missing)
    postings = pd.DataFrame({
        "AFF": round_df.get("AFF", pd.Series([""]*len(round_df))),
        "NEG": round_df.get("NEG", pd.Series([""]*len(round_df))),
        "ROOM": round_df.get("ROOM", pd.Series([""]*len(round_df))).replace("", "TBD")
    })

    # Download CSV
    csv_buf = StringIO()
    postings.to_csv(csv_buf, index=False)
    st.download_button(
        label="⬇️ Download Postings (CSV)",
        data=csv_buf.getvalue(),
        file_name=f"round_{round_number}_postings.csv",
        mime="text/csv",
        key=f"dl_postings_r{round_number}"
    )

    # Pretty print view with a Print button (browser print dialog)
    html_rows = "\n".join(
        f"<tr><td>{a}</td><td>{n}</td><td>{r}</td></tr>"
        for a, n, r in postings[["AFF","NEG","ROOM"]].itertuples(index=False)
    )

    html = f"""
    <div style="font-family: ui-sans-serif, system-ui, -apple-system; max-width: 780px; margin: 1rem auto;">
      <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
        <h3 style="margin:0;">Round {round_number} — Postings</h3>
        <button onclick="window.print()" style="
          padding:8px 12px; border:1px solid #ddd; border-radius:8px; background:#f6f6f6; cursor:pointer;
        ">🖨️ Print</button>
      </div>
      <table style="width:100%; border-collapse:collapse; font-size:15px;">
        <thead>
          <tr>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px 6px;">AFF</th>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px 6px;">NEG</th>
            <th style="text-align:left; border-bottom:1px solid #ddd; padding:8px 6px;">ROOM</th>
          </tr>
        </thead>
        <tbody>
          {html_rows}
        </tbody>
      </table>
      <style>
        @media print {{
          button {{ display: none; }}
          h3 {{ margin-bottom: 8px; }}
          table {{ font-size: 13px; }}
          th, td {{ padding: 6px 5px; }}
        }}
      </style>
    </div>
    """
    st.components.v1.html(html, height=min(400, 80 + 28 * len(postings)))

# ---------------- Normalization helpers ----------------
def _norm_text(s: str) -> str:
    """Lowercase, strip punctuation except spaces, collapse spaces."""
    if s is None:
        return ""
    s = str(s)
    # remove periods and extra punctuation that often sneaks in
    s = s.replace(".", " ").replace(";", " ").replace(":", " ")
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    return s.strip().lower()

def _canonical_name_variants(full_name: str) -> set:
    """
    From 'First Last' make:
      - 'first last'
      - 'last, first'
    From 'Last, First' make:
      - 'first last'
      - 'last, first'
    Return both normalized variants so we can match messy text reliably.
    """
    s = str(full_name).strip()
    if not s:
        return set()
    # If it's "Last, First"
    if "," in s:
        last, first = [t.strip() for t in s.split(",", 1)]
        v1 = _norm_text(f"{first} {last}")
        v2 = _norm_text(f"{last}, {first}")
        return {v1, v2}
    else:
        # Assume "First Last"
        parts = s.split()
        if len(parts) >= 2:
            first = parts[0]
            last = " ".join(parts[1:])
            v1 = _norm_text(f"{first} {last}")
            v2 = _norm_text(f"{last}, {first}")
            return {v1, v2, _norm_text(s)}
        return {_norm_text(s)}

def _parse_listish(cell):
    """
    Try literal-eval for pythonic lists. If it’s a real list/tuple/set already, return list.
    Else return None so caller can do fuzzy extraction.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    if isinstance(cell, (list, tuple, set)):
        return [str(x).strip() for x in cell if str(x).strip()]
    s = str(cell).strip()
    if not s:
        return None
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val if str(x).strip()]
        # single scalar -> single-item list
        return [str(val).strip()]
    except Exception:
        return None

# ---------------- Parse "Cannot Judge" robustly ----------------
def _extract_cannot_judge(cell, roster: list) -> set:
    """
    Returns the actual roster names that this cell forbids.
    Strategy:
      1) If it's a proper list: match list elements to roster via normalized variants.
      2) Otherwise, scan the entire string against every student’s variants.
    """
    # Map of each roster student to its normalized variants
    roster_variants = {student: _canonical_name_variants(student) for student in roster}

    parsed = _parse_listish(cell)
    if parsed is not None:
        # Direct list case
        targets = set()
        parsed_norm = [_norm_text(x) for x in parsed]
        for student, variants in roster_variants.items():
            if any(v in parsed_norm for v in variants):
                targets.add(student)
        return targets

    # Fallback: scan text
    text_norm = _norm_text(str(cell or ""))
    targets = set()
    for student, variants in roster_variants.items():
        # if ANY variant of the student is substring of the normalized text -> forbid
        if any(v and v in text_norm for v in variants):
            targets.add(student)
    return targets

# ---------------- Judging rounds parsing ----------------
def _parse_rounds_signed(cell) -> set:
    """
    Parse 'Judging Rounds' (supports list, stringified list, CSV, scalar).
    Always returns a set of strings, e.g. {'All','1','2'}.
    """
    parsed = _parse_listish(cell)
    if parsed is not None:
        return {str(x).strip() for x in parsed if str(x).strip()}
    s = str(cell or "").strip()
    if not s:
        return {"All"}
    if "," in s:
        return {x.strip() for x in s.split(",")}
    return {s}

# ---------------- Eligibility checks ----------------
def _judge_can_cover(judges_df: pd.DataFrame, judge_name: str, aff: str, neg: str,
                     round_number: int, roster: list, debug=False):
    """
    Returns (eligible: bool, reason: str)
    - Checks Cannot Judge (robust parsing)
    - Checks Judging Rounds contains 'All' or this round number
    """
    if not judge_name:
        return False, "empty-name"

    row = judges_df.loc[judges_df["Full Name"] == judge_name]
    if row.empty:
        return False, "no-row"

    row = row.iloc[0]
    cannot_set = _extract_cannot_judge(row.get("Cannot Judge", ""), roster)
    rounds_set = _parse_rounds_signed(row.get("Judging Rounds", "All"))

    if aff in cannot_set or neg in cannot_set:
        if debug:
            return False, f"cannot-judge: {aff if aff in cannot_set else ''} {neg if neg in cannot_set else ''}".strip()
        return False, "cannot-judge"

    if "All" not in rounds_set and str(round_number) not in rounds_set:
        return False, f"not-signed-for-round-{round_number}"

    return True, "ok"

# ---------------- Assign judges greedily (one per debate) ----------------
def _assign_judges(display_round: pd.DataFrame, judges_df: pd.DataFrame, round_number: int) -> pd.DataFrame:
    """
    Assign one judge per debate while respecting:
      • judge's 'Cannot Judge' list,
      • judge's 'Judging Rounds' contains round_number or 'All',
      • no double-assignments within the same round,
      • after assignment, add both debaters to judge's Cannot Judge to prevent future repeats.

    Mutates st.session_state["judges"] with updated 'Cannot Judge' lists.
    Leaves JUDGE blank if no eligible judge remains for a debate.
    """

    # ---- Helpers -----------------------------------------------------------
    def _parse_listish(cell) -> list[str]:
        """Parse a cell that might be a list, a stringified list, CSV, or empty into a list of strings."""
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            return []
        if isinstance(cell, (list, tuple, set)):
            return [str(x).strip() for x in cell if str(x).strip()]
        s = str(cell).strip()
        if not s:
            return []
        # Try literal list/tuple first (e.g. "['A','B']")
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple, set)):
                return [str(x).strip() for x in val if str(x).strip()]
        except Exception:
            pass
        # Fallback: split on commas/semicolons
        parts = []
        for chunk in s.replace(";", ",").split(","):
            t = chunk.strip()
            if t:
                parts.append(t)
        return parts

    def _judge_can_cover_row(jname: str, aff: str, neg: str, allowed_rounds: list[str], cannot_list: set[str]) -> bool:
        if not jname:
            return False
        # Round permission
        if "All" not in allowed_rounds and str(round_number) not in allowed_rounds:
            return False
        # Cannot Judge
        if aff and aff in cannot_list: 
            return False
        if neg and neg in cannot_list:
            return False
        return True
    # -----------------------------------------------------------------------

    # Ensure JUDGE column exists
    if "JUDGE" not in display_round.columns:
        display_round["JUDGE"] = ""

    # No judges available
    if judges_df is None or judges_df.empty or "Full Name" not in judges_df.columns:
        display_round["JUDGE"] = ""
        return display_round

    # Build a normalized, easy-to-query structure of judges
    # Each judge: name -> {"cannot": set(...), "rounds": list[str]}
    judge_pool = []
    for _, jrow in judges_df.iterrows():
        jname = str(jrow.get("Full Name", "")).strip()
        if not jname:
            continue
        cannot = set(_parse_listish(jrow.get("Cannot Judge", "")))
        rounds_ok = _parse_listish(jrow.get("Judging Rounds", ["All"]))
        if not rounds_ok:
            rounds_ok = ["All"]
        judge_pool.append({"name": jname, "cannot": cannot, "rounds": rounds_ok})

    # Track judges already used in this round to avoid double-assignments
    used_this_round = set()

    # Assign for each debate row
    for idx, row in display_round.iterrows():
        aff = str(row.get("AFF", "")).strip()
        neg = str(row.get("NEG", "")).strip()

        # If there's a BYE, don't assign a judge
        if aff == "BYE" or neg == "BYE":
            display_round.at[idx, "JUDGE"] = ""
            continue

        shuffled = judge_pool[:]  # shallow copy
        random.shuffle(shuffled)

        assigned_name = ""
        for j in shuffled:
            name = j["name"]
            if name in used_this_round:
                continue
            if _judge_can_cover_row(name, aff, neg, j["rounds"], j["cannot"]):
                assigned_name = name
                used_this_round.add(name)
                # Update in-memory cannot list so this judge won't see these students again later
                j["cannot"].update({aff, neg})
                # Also persist back into judges_df so future calls see this immediately
                m = judges_df["Full Name"].astype(str).str.strip() == name
                if m.any():
                    new_cannot = sorted(j["cannot"])
                    judges_df.loc[m, "Cannot Judge"] = "; ".join(new_cannot)
                break

        display_round.at[idx, "JUDGE"] = assigned_name

    # Persist updated judges back to session_state
    st.session_state["judges"] = judges_df

    return display_round
# ==========================
# Admin: Reset tournament
# ==========================
def reset_tournament():
    st.session_state["rounds"] = pd.DataFrame(columns=[
        "Round", "Student", "Opponent", "Side", "Win", "Speaker Points", "Room", "Judge"
    ])
    if "students" in st.session_state:
        df = st.session_state["students"].copy()
        df["Wins"] = 0
        df["Losses"] = 0
        df["Average Speaker Points"] = 0.0
        st.session_state["students"] = df
        save_students()
    st.success("Tournament reset: rounds cleared and standings set to zero.")
    st.rerun()

# ==========================
# Standings recompute
# ==========================
def update_student_records_from_rounds():
    """Recalculate wins, losses, and average speaks from ALL committed (finalized) rounds."""
    load_students()
    load_rounds()

    students_df = st.session_state["students"].copy()
    rounds_df = st.session_state["rounds"].copy()

    # Start everyone at zero
    students_df["Wins"] = 0
    students_df["Losses"] = 0
    students_df["Average Speaker Points"] = 0.0

    if rounds_df.empty:
        st.session_state["students"] = students_df
        save_students()
        return

    speaks_totals = {s: [] for s in students_df["Full Name"]}

    for _, r in rounds_df.iterrows():
        student = r.get("Student", "")
        if student not in speaks_totals:
            continue

        # Speaker points
        try:
            sp = float(r.get("Speaker Points", 0))
        except (ValueError, TypeError):
            sp = 0.0
        speaks_totals[student].append(sp)

        # Wins/Losses logic
        win = str(r.get("Win", "")).upper()
        side = str(r.get("Side", "")).upper()

        if win in ("AFF", "NEG"):
            if win == side:
                # their side won
                students_df.loc[students_df["Full Name"] == student, "Wins"] += 1
            else:
                # their side lost
                students_df.loc[students_df["Full Name"] == student, "Losses"] += 1
        elif win == "DOUBLE LOSS":
            students_df.loc[students_df["Full Name"] == student, "Losses"] += 1
        # if win == "" (no decision) -> do nothing

    # Compute average speaks
    for s, speaks in speaks_totals.items():
        avg = round(sum(speaks) / len(speaks), 2) if speaks else 0.0
        students_df.loc[students_df["Full Name"] == s, "Average Speaker Points"] = avg

    # Sort and persist
    students_df = students_df.sort_values(
        by=["Wins", "Average Speaker Points"], ascending=[False, False]
    ).reset_index(drop=True)

    st.session_state["students"] = students_df
    save_students()

# ==========================
# Judge / Room helpers
# ==========================
def _norm(s: str) -> str:
    return " ".join(str(s or "").strip().lower().split())

def _name_variants(full_name: str) -> set:
    """
    Return a set of normalized variants for a single person name.
    Accepts either 'First Last' or 'Last, First' style and creates both variants.
    """
    s = str(full_name or "").strip()
    if not s:
        return set()

    s_norm = _norm(s)
    variants = set()

    # If it looks like "Last, First"
    if "," in s:
        parts = [p.strip() for p in s.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            last, first = parts[0], parts[1]
            variants.add(_norm(f"{first} {last}"))       # "first last"
            variants.add(_norm(f"{last}, {first}"))      # "last, first"
            variants.add(_norm(f"{last},{first}"))       # "last,first"
    else:
        # Assume "First Last"
        parts = s.split()
        if len(parts) >= 2:
            first = parts[0]
            last = " ".join(parts[1:])
            variants.add(_norm(f"{first} {last}"))       # "first last"
            variants.add(_norm(f"{last}, {first}"))      # "last, first"
            variants.add(_norm(f"{last},{first}"))       # "last,first"
        else:
            # Single token fallback
            variants.add(s_norm)

    return variants

def _flatten_names_to_set(items) -> set:
    """
    Given an iterable of name strings, return a flat set of all normalized variants
    for *each* item. This lets us check membership robustly later.
    """
    out = set()
    for it in (items or []):
        out.update(_name_variants(it))
    return out

# ---------- Parsing helpers ----------
def _smart_split_names(raw: str) -> list:
    """
    Robustly split a string that may contain:
      - Semicolon-delimited names: "Doe, Jane; Smith, John"
      - Pipe-delimited names: "Doe, Jane | Smith, John"
      - CSV with commas in names: "Doe, Jane, Smith, John"  -> pairs into ["Doe, Jane", "Smith, John"]
      - Simple CSV of "First Last" names.
    Returns a list of clean name strings.
    """
    s = str(raw or "").strip()
    if not s:
        return []

    # Prefer explicit delimiters first
    if ";" in s:
        parts = [p.strip() for p in s.split(";") if p.strip()]
        return parts
    if "|" in s:
        parts = [p.strip() for p in s.split("|") if p.strip()]
        return parts

    # If it looks like a Python list, try to parse it
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, set)):
            return [str(x).strip() for x in val if str(x).strip()]
    except Exception:
        pass

    # Otherwise, we have a CSV-ish string. It may be:
    #  - "First Last, Another Name"
    #  - or "Last, First, Last2, First2" -> need to group into pairs.
    # Heuristic: if we detect multiple commas and they alternate "Last, First" pattern,
    # chunk tokens in pairs.
    tokens = [t.strip() for t in s.split(",") if t.strip()]
    if len(tokens) >= 4:
        # Try to pair as (last, first) repeatedly
        paired = []
        i = 0
        while i + 1 < len(tokens):
            # Combine as "Last, First"
            paired.append(f"{tokens[i]}, {tokens[i+1]}")
            i += 2
        # If odd leftover, keep as-is
        if i < len(tokens):
            paired.append(tokens[i])
        return paired

    # Fallback: a simple comma-separated list of names without internal commas
    return [p.strip() for p in s.split(",") if p.strip()]

def _parse_cannot_judge_names(cell) -> list:
    """
    Parse the 'Cannot Judge' field into a list of name strings,
    handling lists, CSV, 'Last, First' patterns, semicolons, pipes, etc.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    if isinstance(cell, (list, tuple, set)):
        return [str(x).strip() for x in cell if str(x).strip()]
    return _smart_split_names(str(cell))

def _parse_rounds_field(cell) -> set:
    """
    Parse 'Judging Rounds' into a set of strings like {'All','1','2'}.
    Accept list types or delimited strings.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return {"All"}
    if isinstance(cell, (list, tuple, set)):
        return {str(x).strip() for x in cell if str(x).strip()}
    s = str(cell).strip()
    # Try list literal first
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple, set)):
            return {str(x).strip() for x in val if str(x).strip()}
    except Exception:
        pass
    # Otherwise split by common delimiters
    for delim in [";", "|"]:
        if delim in s:
            return {x.strip() for x in s.split(delim) if x.strip()}
    # CSV fallback
    return {x.strip() for x in s.split(",") if x.strip()}

# ---------- Eligibility check ----------
def _judge_can_cover_round(judges_df: pd.DataFrame, judge_name: str,
                           aff: str, neg: str, round_number: int) -> bool:
    if not judge_name:
        return False

    mask = judges_df["Full Name"].astype(str).str.strip() == str(judge_name).strip()
    if not mask.any():
        return False

    row = judges_df.loc[mask].iloc[0]

    # Rounds
    rounds_ok = _parse_rounds_field(row.get("Judging Rounds", "All"))
    if ("All" not in rounds_ok) and (str(round_number) not in rounds_ok):
        return False

    # Cannot Judge list (robust parsing)
    cannot_names = _parse_cannot_judge_names(row.get("Cannot Judge", ""))
    cannot_variants = _flatten_names_to_set(cannot_names)

    # Compare using variants
    aff_variants = _name_variants(aff)
    neg_variants = _name_variants(neg)

    # If any variant matches, judge cannot cover
    if aff_variants & cannot_variants:
        return False
    if neg_variants & cannot_variants:
        return False

    return True

# ---------- Assignment ----------
def _assign_judges(display_round: pd.DataFrame, judges_df: pd.DataFrame,
                   round_number: int) -> pd.DataFrame:
    """
    Assign one judge per debate while respecting:
      - Cannot Judge list,
      - Judging Rounds contains round_number or 'All',
      - No double-assignments in the same round.
    Leaves blank if no eligible judge remains.
    Skips BYE debates.
    """
    # Ensure JUDGE column exists
    if "JUDGE" not in display_round.columns:
        display_round["JUDGE"] = ""
    else:
        display_round["JUDGE"] = display_round["JUDGE"].fillna("")

    if judges_df is None or judges_df.empty:
        return display_round

    judge_pool = judges_df["Full Name"].dropna().astype(str).str.strip().tolist()
    used = set()

    for idx, row in display_round.iterrows():
        aff = str(row.get("AFF", "")).strip()
        neg = str(row.get("NEG", "")).strip()

        # No judge for BYE debates
        if aff.upper() == "BYE" or neg.upper() == "BYE":
            display_round.at[idx, "JUDGE"] = ""
            continue

        shuffled = judge_pool[:]
        random.shuffle(shuffled)

        assigned = ""
        for j in shuffled:
            if j in used:
                continue
            if _judge_can_cover_round(judges_df, j, aff, neg, round_number):
                assigned = j
                used.add(j)
                break

        display_round.at[idx, "JUDGE"] = assigned  # empty if none

    return display_round

def _eligible_rooms_for_round(rooms_df: pd.DataFrame, round_number: int) -> list:
    """Return list of room names where Rounds Available contains round_number or 'All'."""
    if rooms_df is None or rooms_df.empty:
        return []

    eligible = []
    for _, r in rooms_df.iterrows():
        rv = r.get("Rounds Available", "All")
        try:
            if pd.isna(rv).item() if hasattr(pd.isna(rv), "item") else pd.isna(rv):
                rounds_available = {"All"}
            else:
                s = str(rv).strip()
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple, set)):
                        rounds_available = {str(x).strip() for x in parsed}
                    else:
                        rounds_available = {str(parsed).strip()}
                except Exception:
                    if "," in s:
                        rounds_available = {x.strip() for x in s.split(",")}
                    else:
                        rounds_available = {s}
        except Exception:
            rounds_available = {"All"}

        if "All" in rounds_available or str(round_number) in rounds_available:
            room_name = r.get("Room Number", "")
            if room_name:
                eligible.append(room_name)

    return eligible

def _assign_rooms(display_round: pd.DataFrame, rooms_df: pd.DataFrame, round_number: int) -> pd.DataFrame:
    """One room per debate; respect 'Rounds Available'; blank if not enough rooms."""
    rooms = _eligible_rooms_for_round(rooms_df, round_number) if rooms_df is not None else []
    random.shuffle(rooms)
    assigned = []
    for _ in range(len(display_round)):
        assigned.append(rooms.pop() if rooms else "")
    display_round["ROOM"] = assigned
    return display_round

# ==========================
# Round generation helpers
# ==========================
def default_generate_matchups(rn: int) -> pd.DataFrame:
    """
    Simple random pairings with BYE if odd.
    Returns columns: ["Round","AFF","NEG"] for display.
    """
    students_df = st.session_state.get("students", pd.DataFrame(columns=["Full Name"]))
    names = students_df.get("Full Name", pd.Series(dtype=str)).dropna().tolist()
    random.shuffle(names)
    rows = []
    for i in range(0, len(names) - 1, 2):
        rows.append({"Round": rn, "AFF": names[i], "NEG": names[i + 1]})
    if len(names) % 2 == 1:
        rows.append({"Round": rn, "AFF": names[-1], "NEG": "BYE"})
    df = pd.DataFrame(rows, columns=["Round", "AFF", "NEG"])
    # Editor fields
    for c in ["JUDGE", "ROOM", "WIN", "AFF SPEAKS", "NEG SPEAKS"]:
        df[c] = ""
    return df

def default_prepare_aff_neg_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pass-through if already AFF/NEG schema.
    Otherwise try to pivot from side-based rows.
    """
    if {"AFF", "NEG", "Round"}.issubset(raw_df.columns):
        out = raw_df[["Round", "AFF", "NEG"]].copy()
    else:
        need = {"Round", "Student", "Opponent", "Side"}
        if need.issubset(raw_df.columns):
            aff_rows = raw_df[raw_df["Side"].str.upper() == "AFF"].rename(
                columns={"Student": "AFF", "Opponent": "NEG"}
            )
            out = aff_rows[["Round", "AFF", "NEG"]].drop_duplicates().copy()
        else:
            out = pd.DataFrame(columns=["Round", "AFF", "NEG"])
    for c in ["JUDGE", "ROOM", "WIN", "AFF SPEAKS", "NEG SPEAKS"]:
        if c not in out.columns:
            out[c] = ""
    return out[["Round", "AFF", "NEG", "JUDGE", "ROOM", "WIN", "AFF SPEAKS", "NEG SPEAKS"]]

# ==========================
# Tabs: Tournament Hub / Judges / Rooms / Schedule
# ==========================
def tournament_hub():
    st.title("🏆 Tournament Hub")

    students_df = load_students()

    # Add a student
    with st.form("add_student", clear_on_submit=True):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        school = st.text_input("School")
        submitted = st.form_submit_button("Add Student")

        if submitted:
            if not first_name.strip() or not last_name.strip():
                st.error("Please enter both first and last name")
            else:
                full_name = f"{last_name.strip()}, {first_name.strip()}"
                new_row = {
                    "First Name": first_name.strip(),
                    "Last Name": last_name.strip(),
                    "Full Name": full_name,
                    "School": school.strip(),
                    "Wins": 0,
                    "Losses": 0,
                    "Average Speaker Points": 0.0
                }
                st.session_state["students"] = pd.concat(
                    [st.session_state["students"], pd.DataFrame([new_row])],
                    ignore_index=True
                )
                save_students()
                st.success(f"Added {full_name}")

    st.subheader("📊 Current Standings")
    students_df = st.session_state["students"].copy()
    if students_df.empty:
        st.info("No students added yet.")
    else:
        students_df = students_df.sort_values(
            by=["Wins", "Average Speaker Points"], ascending=[False, False]
        ).reset_index(drop=True)
        st.dataframe(students_df, use_container_width=True)

    st.subheader("Remove a Student")
    if not students_df.empty:
        student_to_remove = st.selectbox(
            "Select a student to remove",
            options=[""] + students_df["Full Name"].tolist(),
            index=0,
            key="remove_student_select"
        )
        if st.button("Remove Student", key="remove_student_btn"):
            if student_to_remove:
                st.session_state["students"] = students_df[
                    students_df["Full Name"] != student_to_remove
                ].reset_index(drop=True)
                save_students()
                st.success(f"Removed {student_to_remove}")
            else:
                st.warning("Please select a student to remove.")
    else:
        st.info("No students available to remove.")

    with st.expander("Admin"):
        if st.button("🔄 Reset Tournament (clear rounds & zero standings)"):
            reset_tournament()


def judges_tab():
    st.header("⚖️ Judges Pool")
    students_df = st.session_state.get("students", pd.DataFrame())
    student_names = students_df["Full Name"].tolist() if not students_df.empty else []

    judges_df = load_judges()

    # Add Judge
    with st.form("add_judge", clear_on_submit=True):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        cannot_judge = st.multiselect(
            "Cannot Judge (Select students)",
            options=student_names
        )
        rounds_judging = st.multiselect(
            "Judging Rounds (Select rounds you plan to judge)",
            options=('1', '2', '3', '4', 'Semifinals', 'Finals')
        )
        submitted = st.form_submit_button("Add Judge")

        if submitted:
            if not first_name.strip() or not last_name.strip():
                st.error("Please enter both first and last name")
            else:
                full_name = f"{last_name.strip()}, {first_name.strip()}"
                cannot_judge_str = ", ".join(cannot_judge)
                new_row = {
                    "First Name": first_name.strip(),
                    "Last Name": last_name.strip(),
                    "Full Name": full_name,
                    "Cannot Judge": cannot_judge_str,
                    "Judging Rounds": rounds_judging
                }
                st.session_state["judges"] = pd.concat(
                    [st.session_state["judges"], pd.DataFrame([new_row])],
                    ignore_index=True
                )
                st.success(f"Added Judge: {full_name}")

    st.subheader("All Judges")
    judges_df = st.session_state["judges"]
    if judges_df.empty:
        st.info("No judges added yet.")
    else:
        st.dataframe(judges_df, use_container_width=True)

    st.subheader("Remove a Judge")
    judges_df = st.session_state["judges"]
    if not judges_df.empty:
        judge_to_remove = st.selectbox(
            "Select a judge to remove",
            options=[""] + judges_df["Full Name"].tolist(),
            index=0,
            key="remove_judge_select"
        )
        if st.button("Remove Judge", key="remove_judge_btn"):
            if judge_to_remove:
                st.session_state["judges"] = judges_df[
                    judges_df["Full Name"] != judge_to_remove
                ].reset_index(drop=True)
                st.success(f"Removed {judge_to_remove}")
            else:
                st.warning("Please select a judge to remove.")
    else:
        st.info("No judges available to remove.")


def rooms_tab():
    st.header("🏛️ Available Rooms")
    rooms_df = load_rooms()

    with st.form("add_room", clear_on_submit=True):
        room_num = st.text_input("Room Number")
        round_available = st.multiselect(
            "Available Rounds (Select rounds this room is available for)",
            options=('All', '1', '2', '3', '4', 'Semifinals', 'Finals')
        )
        submitted = st.form_submit_button("Add Room")

        if submitted:
            if not room_num.strip() or not round_available:
                st.error("Please enter the room number and available rounds")
            else:
                new_row = {
                    "Room Number": room_num.strip(),
                    "Rounds Available": round_available
                }
                st.session_state["rooms"] = pd.concat(
                    [st.session_state["rooms"], pd.DataFrame([new_row])],
                    ignore_index=True
                )
                save_rooms()
                st.success(f"Added {room_num}")

    st.subheader("All Rooms")
    rooms_df = st.session_state["rooms"].copy()
    if rooms_df.empty:
        st.info("No rooms added yet.")
    else:
        st.dataframe(rooms_df, use_container_width=True)

    st.subheader("Remove a room")
    rooms_df = st.session_state["rooms"]
    if not rooms_df.empty:
        rooms_to_remove = st.selectbox(
            "Select a room to remove",
            options=[""] + rooms_df["Room Number"].tolist(),
            index=0,
            key="remove_room_select"
        )
        if st.button("Remove room", key="remove_room_btn"):
            if rooms_to_remove:
                st.session_state["rooms"] = rooms_df[
                    rooms_df["Room Number"] != rooms_to_remove
                ].reset_index(drop=True)
                save_rooms()
                st.success(f"Removed {rooms_to_remove}")
            else:
                st.warning("Please select a room to remove.")
    else:
        st.info("No rooms available to remove.")


def schedule_tab():
    st.header("🕰️ Tournament Schedule")
    st.markdown("<p style='font-size:14px; font-weight:normal; margin-bottom:5px;'>Update scheduling to display on round postings</p>", unsafe_allow_html=True)

    schedule_df = load_schedule()

    for i, row in schedule_df.iterrows():
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            inner_col1, inner_col2 = st.columns([1, 1])
            with inner_col1:
                st.write(f"**{row['Round']}**")
            with inner_col2:
                time_str = str(row["Time"]).strip()
                if not time_str:
                    default_time = time(9, 0)
                else:
                    t_part, ampm = time_str[:-3], time_str[-2:]
                    h_str, m_str = t_part.split(":")
                    h, m = int(h_str), int(m_str)
                    if ampm.upper() == "PM" and h != 12:
                        h += 12
                    elif ampm.upper() == "AM" and h == 12:
                        h = 0
                    default_time = time(h, m)

                st.session_state["schedule"].at[i, "Time"] = st.time_input(
                    label="",
                    value=default_time,
                    key=f"round_time_{i}",
                    label_visibility="collapsed"
                ).strftime("%I:%M %p")

    st.subheader("📋 Current Schedule")
    st.dataframe(st.session_state["schedule"], use_container_width=True)

# ==========================
# Round Tab (stable editor pattern)
# ==========================
def round_tab(round_number: int):
    """
    Guarantees, across Rounds 1–4:
      • Each student has exactly 2 AFF and 2 NEG total (BYEs count toward the side).
      • Exactly 2 AFF-BYEs and 2 NEG-BYEs overall.
    For Rounds 3–4:
      • Power-pair inside wins brackets: high avg speaks vs low avg speaks within each bracket.
      • If a bracket is odd/misaligned, float the lowest to pair with the highest in the next bracket.
    For all rounds:
      • Avoid rematches; if stuck, relax minimally.
    Produces/edits a display table with:
      ["Round","AFF","NEG","JUDGE","ROOM","WIN","AFF SPEAKS","NEG SPEAKS"]
    """
    import ast
    st.header(f"✒️ Round {round_number}")

    # ---------------- small utils ----------------
    def _as_df(name, cols):
        df = st.session_state.get(name)
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(columns=cols)
            st.session_state[name] = df
        return df

    def _finalized():
        return st.session_state.get("rounds", pd.DataFrame(columns=[
            "Round","Student","Opponent","Side","Win","Speaker Points","Room","Judge"
        ]))

    def _display_key():
        return f"round_{round_number}"

    round_key  = _display_key()
    orig_key   = f"{round_key}_orig"
    edited_key = f"{round_key}_edited"
    editor_widget_key = f"editor_widget_{round_number}"

    # Ensure display store exists
    _as_df(round_key, ["Round","AFF","NEG","JUDGE","ROOM","WIN","AFF SPEAKS","NEG SPEAKS"])

    # ---------- Standing/usage readers ----------
    def _side_counts_all() -> dict:
        """
        Return {student: {"AFF":a,"NEG":n,"BYE":b}} from finalized table only.
        """
        rr = _finalized()
        out = {}
        if rr.empty: return out
        for _, r in rr.iterrows():
            s   = str(r.get("Student","")).strip()
            opp = str(r.get("Opponent","")).strip()
            side= str(r.get("Side","")).strip().upper()
            if not s: continue
            d = out.setdefault(s, {"AFF":0,"NEG":0,"BYE":0})
            if opp == "BYE":
                d["BYE"] += 1
                if side in ("AFF","NEG"): d[side] += 1
            else:
                if side in ("AFF","NEG"): d[side] += 1
        return out

    def _wins_and_avg():
        stu = st.session_state.get("students", pd.DataFrame(columns=["Full Name","Wins","Average Speaker Points"]))
        if stu.empty:
            return {}, {}
        wins = dict(zip(stu["Full Name"], stu["Wins"].fillna(0)))
        avg  = dict(zip(stu["Full Name"], stu["Average Speaker Points"].fillna(0.0)))
        return wins, avg

    def _global_bye_side_counts() -> dict:
        """Total BYEs by side from finalized + any unfinalized round_* display tables."""
        totals = {"AFF":0,"NEG":0}
        # finalized
        rr = _finalized()
        if not rr.empty:
            mask = rr["Opponent"].astype(str) == "BYE"
            if mask.any():
                side_counts = rr.loc[mask, "Side"].str.upper().value_counts()
                totals["AFF"] += int(side_counts.get("AFF",0))
                totals["NEG"] += int(side_counts.get("NEG",0))
        # unfinalized displays
        for k, v in st.session_state.items():
            if isinstance(k, str) and k.startswith("round_") and isinstance(v, pd.DataFrame) and not v.empty:
                if {"AFF","NEG"}.issubset(v.columns):
                    totals["AFF"] += int((v["NEG"].astype(str) == "BYE").sum())
                    totals["NEG"] += int((v["AFF"].astype(str) == "BYE").sum())
        return totals

    def _avoid_pairs() -> set:
        rr = _finalized()
        seen = set()
        if rr.empty: return seen
        for _, r in rr.iterrows():
            s1, s2 = str(r["Student"]).strip(), str(r["Opponent"]).strip()
            if s1 and s2 and s1 != "BYE" and s2 != "BYE":
                seen.add(tuple(sorted([s1, s2])))
        return seen

    # ---------- Side planning ----------
    def _needed_side(student: str, counts: dict) -> str:
        """Return the side this student needs more, to hit exactly 2/2 at end."""
        c = counts.get(student, {"AFF":0,"NEG":0})
        need_aff = 2 - int(c["AFF"])
        need_neg = 2 - int(c["NEG"])
        if need_aff > need_neg: return "AFF"
        if need_neg > need_aff: return "NEG"
        return "AFF" if random.random() < 0.5 else "NEG"

    def _choose_bye_side(n_students: int) -> str|None:
        if n_students % 2 == 0:
            return None
        totals = _global_bye_side_counts()
        # pick the side that still needs to reach 2
        if totals["AFF"] < 2 and totals["NEG"] < 2:
            return "AFF" if totals["AFF"] < totals["NEG"] else "NEG"
        if totals["AFF"] < 2: return "AFF"
        if totals["NEG"] < 2: return "NEG"
        return "AFF"

    def _rank_bye_candidates(students: list[str], bye_side: str) -> list[str]:
        """Prefer students who still need that bye_side, then lower wins/avg speaks, fewer BYEs."""
        stu = st.session_state.get("students", pd.DataFrame())
        rr  = _finalized()
        # prep wins/speaks
        base = stu.set_index("Full Name")[["Wins","Average Speaker Points"]].reindex(students).fillna(0)
        # bye counts
        bye_count = {s:0 for s in students}
        if not rr.empty:
            mask = rr["Opponent"].astype(str) == "BYE"
            if mask.any():
                for _, r in rr.loc[mask].iterrows():
                    name = str(r["Student"]).strip()
                    if name in bye_count:
                        bye_count[name] += 1
        # side needs
        counts = _side_counts_all()
        needs = {s: _needed_side(s, counts) for s in students}
        # rank
        return sorted(
            students,
            key=lambda s: (
                0 if needs.get(s) == bye_side else 1,             # need this side first
                base.at[s, "Wins"],
                base.at[s, "Average Speaker Points"],
                bye_count.get(s,0)
            )
        )

    def _plan_targets(students: list[str], bye_choice: tuple[str,str]|None) -> dict:
        """
        Return {student: 'AFF'|'NEG'} targets for this round, meeting remaining quotas.
        Balance the two bins as evenly as possible.
        """
        counts = _side_counts_all()
        pool = students[:]
        if bye_choice is not None:
            _side, bye_student = bye_choice
            if bye_student in pool:
                pool.remove(bye_student)

        targets = {s: _needed_side(s, counts) for s in pool}
        affs = [s for s in pool if targets[s] == "AFF"]
        negs = [s for s in pool if targets[s] == "NEG"]

        # balance bins
        def _shift(src, dst, new_side):
            while len(src) > len(dst):
                s = src.pop()
                targets[s] = new_side
                dst.append(s)
        _shift(affs, negs, "NEG")
        _shift(negs, affs, "AFF")
        return targets

    # ---------- NEW: power pairing for R3–R4 inside pairer ----------
    def _pair_with_targets(students: list[str], targets: dict, avoid: set, bye_choice: tuple[str,str]|None):
        """
        Pair students so that each student goes on their target side.
        R1–R2: simple cross-bin greedy with rematch avoidance.
        R3–R4: power-pair per wins bracket (high avg vs low avg),
               float leftovers to adjacent brackets, still honoring targets.
        """
        pool = students[:]
        if bye_choice:
            _, bye_student = bye_choice
            if bye_student in pool: pool.remove(bye_student)

        wins_map, avg_map = _wins_and_avg()

        def _simple_pair():
            A = [s for s in pool if targets.get(s) == "AFF"]
            N = [s for s in pool if targets.get(s) == "NEG"]
            random.shuffle(A); random.shuffle(N)
            pairs, used_n = [], set()
            for a in A:
                partner = None
                for n in N:
                    if n in used_n: continue
                    if tuple(sorted([a, n])) in avoid: continue
                    partner = n; break
                if partner is None:
                    for n in N:
                        if n not in used_n:
                            partner = n; break
                if partner:
                    used_n.add(partner)
                    pairs.append((a, partner))
            return pairs

        if round_number in (1, 2):
            pairs = _simple_pair()
        else:
            # ---- R3–R4: power pairing by wins bracket ----
            # build brackets: wins -> list of students, sorted by avg speaks desc (high first)
            brackets = {}
            for s in pool:
                w = int(wins_map.get(s, 0))
                brackets.setdefault(w, []).append(s)
            for w in brackets:
                brackets[w].sort(key=lambda s: float(avg_map.get(s,0.0)), reverse=True)

            # For each bracket, separate by target side
            leftovers_aff, leftovers_neg = [], []
            pairs = []

            sorted_wins = sorted(brackets.keys(), reverse=True)  # top bracket first
            for w in sorted_wins:
                group = brackets[w]
                affs = [s for s in group if targets.get(s) == "AFF"]
                negs = [s for s in group if targets.get(s) == "NEG"]

                # high-vs-low: sort affs high->low, negs high->low, then pick neg from the bottom
                affs.sort(key=lambda s: float(avg_map.get(s,0.0)), reverse=True)
                negs.sort(key=lambda s: float(avg_map.get(s,0.0)), reverse=True)

                used_neg = set()
                for a in affs:
                    # try lowest-speaking NEG first to get high-vs-low
                    partner = None
                    for n in reversed(negs):
                        if n in used_neg: continue
                        if tuple(sorted([a, n])) in avoid: continue
                        partner = n; break
                    if partner is None:
                        # relax rematch if needed
                        for n in reversed(negs):
                            if n not in used_neg:
                                partner = n; break
                    if partner:
                        used_neg.add(partner)
                        pairs.append((a, partner))

                # leftovers in this bracket (unpaired because of side imbalance)
                remaining_aff = [s for s in affs if all(s not in p for p in pairs)]
                remaining_neg = [s for s in negs if all(s not in p for p in pairs)]
                leftovers_aff.extend(remaining_aff)
                leftovers_neg.extend(remaining_neg)

            # Cross-bracket float: lowest of leftovers_neg vs highest of leftovers_aff, and vice versa
            leftovers_aff.sort(key=lambda s: float(avg_map.get(s,0.0)), reverse=True)  # high first
            leftovers_neg.sort(key=lambda s: float(avg_map.get(s,0.0)), reverse=True)  # high first

            used_neg = set()
            for a in leftovers_aff:
                partner = None
                # try from lowest of leftovers_neg
                for n in reversed(leftovers_neg):
                    if n in used_neg: continue
                    if tuple(sorted([a, n])) in avoid: continue
                    partner = n; break
                if partner is None:
                    for n in reversed(leftovers_neg):
                        if n not in used_neg:
                            partner = n; break
                if partner:
                    used_neg.add(partner)
                    pairs.append((a, partner))

            # note: if some still remain due to extreme imbalance, they’ll be handled by BYE or remain unpaired
            # (with BYE already chosen globally)

        # add BYE at the end if any
        if bye_choice:
            side, bye_student = bye_choice
            if side == "AFF":
                pairs.append((bye_student, "BYE"))
            else:
                pairs.append(("BYE", bye_student))

        return pairs

    # ---------------- control bar ----------------
    left, mid, right = st.columns([1.7, 1, 1.2])
    gen_clicked = left.button(f"🔁 Generate / Regenerate Round {round_number}", key=f"gen_btn_{round_number}")
    clr_clicked = right.button("🧹 Clear This Round", key=f"clear_btn_{round_number}")

    if clr_clicked:
        st.session_state[round_key] = st.session_state[round_key].iloc[0:0].copy()
        st.session_state.pop(orig_key, None)
        st.session_state.pop(edited_key, None)
        st.rerun()

    # ---------------- generation ----------------
    if gen_clicked:
        students_df = st.session_state.get("students", pd.DataFrame())
        names = students_df.get("Full Name", pd.Series(dtype=str)).dropna().tolist()

        # decide BYE side & recipient (if odd)
        bye_side = _choose_bye_side(len(names))
        bye_student = None
        if bye_side:
            ranked = _rank_bye_candidates(names, bye_side)
            bye_student = ranked[0] if ranked else None
        bye_choice = (bye_side, bye_student) if (bye_side and bye_student) else None

        # plan targets and pair
        targets = _plan_targets(names, bye_choice)
        avoid   = _avoid_pairs()
        pairs   = _pair_with_targets(names, targets, avoid, bye_choice)

        # Build display DF
        disp = pd.DataFrame({
            "Round": [round_number]*len(pairs),
            "AFF": [a for a,b in pairs],
            "NEG": [b for a,b in pairs],
        })

        # Judges / Rooms (use your existing helpers if defined)
        judges_df = st.session_state.get("judges", pd.DataFrame(columns=["First Name","Last Name","Full Name","Cannot Judge","Judging Rounds"]))
        rooms_df  = st.session_state.get("rooms",  pd.DataFrame(columns=["Room Number","Rounds Available"]))

        # Student roster (list of Full Name values)
        student_roster = st.session_state.get("students", pd.DataFrame()).get("Full Name", pd.Series(dtype=str)).dropna().tolist()


        if "_assign_judges" in globals():
            disp = _assign_judges(disp, judges_df, round_number)
            # st.session_state["judges"] = judges_df
        else:
            disp["JUDGE"] = ""

        if "_assign_rooms" in globals():
            disp = _assign_rooms(disp, rooms_df, round_number)
        else:
            disp["ROOM"] = ""

        # init editable fields as strings
        for c in ["WIN","AFF SPEAKS","NEG SPEAKS"]:
            disp[c] = ""

        st.session_state[round_key] = disp.copy()
        st.session_state[orig_key]  = disp.copy()
        st.session_state[edited_key]= disp.copy()
        st.rerun()

    # ---------------- editor ----------------
    st.subheader("📋 Current Round Matchups")

    if orig_key not in st.session_state:
        st.session_state[orig_key] = st.session_state[round_key].copy()

    df_to_edit = st.session_state[orig_key].copy()
    if df_to_edit.empty:
        st.info("No matchups yet. Click **Generate / Regenerate** above to create this round.")
        return

    for col in ["WIN","AFF SPEAKS","NEG SPEAKS"]:
        if col in df_to_edit:
            df_to_edit[col] = df_to_edit[col].fillna("").astype(str)

    column_config = {
        "WIN": st.column_config.SelectboxColumn("WIN", options=["", "AFF", "NEG", "DOUBLE LOSS"]),
        "AFF SPEAKS": st.column_config.TextColumn("AFF SPEAKS"),
        "NEG SPEAKS": st.column_config.TextColumn("NEG SPEAKS")
    }

    edited_df = st.data_editor(
        df_to_edit,
        key=editor_widget_key,
        column_config=column_config,
        use_container_width=True,
        hide_index=True,
        num_rows="fixed"
    )
    st.session_state[edited_key] = edited_df.copy()

    # ---------------- finalize ----------------
    if st.button(f"✅ Finalize Round {round_number}", key=f"finalize_btn_{round_number}"):
        # commit edits back to the display table
        st.session_state[round_key] = st.session_state[edited_key].copy()
        st.session_state[orig_key]  = st.session_state[edited_key].copy()

        # append to global rounds (side-based)
        _as_df("rounds", ["Round","Student","Opponent","Side","Win","Speaker Points","Room","Judge"])
        out = []
        for _, r in st.session_state[round_key].iterrows():
            out.append({"Round": r["Round"], "Student": r["AFF"], "Opponent": r["NEG"],
                        "Side": "AFF", "Win": r["WIN"], "Speaker Points": r["AFF SPEAKS"],
                        "Room": r["ROOM"], "Judge": r["JUDGE"]})
            out.append({"Round": r["Round"], "Student": r["NEG"], "Opponent": r["AFF"],
                        "Side": "NEG", "Win": r["WIN"], "Speaker Points": r["NEG SPEAKS"],
                        "Room": r["ROOM"], "Judge": r["JUDGE"]})
        if out:
            st.session_state["rounds"] = pd.concat([st.session_state["rounds"], pd.DataFrame(out)], ignore_index=True)

        # standings update (if your helper exists)
        if "update_student_records_from_rounds" in globals():
            update_student_records_from_rounds()

        st.success(f"✅ Round {round_number} finalized. Standings updated.")
        st.rerun()

    st.markdown("### 🗒️ Print / Post Round")
    # Use the latest edited view if present; otherwise the stored display table
    _postings_src = st.session_state.get(f"round_{round_number}_edited",
                    st.session_state.get(f"round_{round_number}", pd.DataFrame()))
    # Make sure the essentials exist
    expected_cols = {"Round","AFF","NEG","ROOM"}
    if isinstance(_postings_src, pd.DataFrame) and expected_cols.issubset(set(_postings_src.columns)):
        render_printable_postings(_postings_src[["Round","AFF","NEG","ROOM"]], round_number)
    else:
        st.info("Generate the round first to view printable postings.")
        

def finals_tab():
    st.header("🏆 Final")

    # Need standings to seed
    if "students" not in st.session_state or st.session_state["students"].empty:
        st.info("Add students / finalize prelim rounds to seed a Final.")
        return

    # ---------- helpers ----------
    def _top2():
        df = st.session_state["students"].copy()
        df = df.sort_values(
            by=["Wins", "Average Speaker Points"],
            ascending=[False, False]
        )
        return df.head(2)[["Full Name"]].reset_index(drop=True)

    def _seed_final(top2_df: pd.DataFrame) -> pd.DataFrame:
        """
        Seed Final: #1 vs #2.
        Columns: Match | AFF | NEG | WIN
        """
        if len(top2_df) < 2:
            return pd.DataFrame(columns=["Match", "AFF", "NEG", "WIN"])
        aff = top2_df.loc[0, "Full Name"]
        neg = top2_df.loc[1, "Full Name"]
        return pd.DataFrame([{"Match": "Final", "AFF": aff, "NEG": neg, "WIN": ""}])

    # ---------- (re)seed controls ----------
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("🔁 Reseed Final from Standings", key="reseat_final_btn"):
            t2 = _top2()
            if len(t2) < 2:
                st.warning("Need at least 2 students to stage a Final.")
            else:
                st.session_state["outrounds_finals"] = _seed_final(t2)

    # Create initial final if needed
    if "outrounds_finals" not in st.session_state or st.session_state["outrounds_finals"].empty:
        t2 = _top2()
        st.session_state["outrounds_finals"] = _seed_final(t2)

    finals_df = st.session_state["outrounds_finals"].copy()

    if finals_df.empty:
        st.info("Not enough competitors to stage a Final.")
        return

    # ---------- edit WIN only (AFF / NEG) ----------
    st.subheader("Final Matchup")
    final_config = {
        "WIN": st.column_config.SelectboxColumn("WIN", options=["", "AFF", "NEG"])
    }
    finals_edited = st.data_editor(
        finals_df,
        key="finals_editor_single",
        column_config=final_config,
        hide_index=True,
        use_container_width=True
    )
    st.session_state["outrounds_finals"] = finals_edited

    # ---------- finalize ----------
    if st.button("🏁 Finalize Final", key="finalize_final_single"):
        r = finals_edited.iloc[0]
        champ = r["AFF"] if r.get("WIN") == "AFF" else (r["NEG"] if r.get("WIN") == "NEG" else "")
        if champ:
            st.success(f"🏆 Champion: **{champ}**")
        else:
            st.warning("Pick a winner (AFF/NEG) in the Final.")
# ==========================
# Main
# ==========================
def main():
    st.title("🏛️ ACA Classical Qualifier")

    tabs = st.tabs(["Tournament Hub", "Judges Pool", "Rooms",
                    "Round One", "Round Two", "Round Three", "Outrounds"])

    with tabs[0]:
        tournament_hub()
    with tabs[1]:
        judges_tab()
    with tabs[2]:
        rooms_tab()
    with tabs[3]:
        round_tab(1)
    with tabs[4]:
        round_tab(2)
    with tabs[5]:
        round_tab(3)
    #with tabs[6]:
    #    round_tab(4)
    with tabs[6]:
        finals_tab()
    # If you also want schedule:
    # with st.tabs(["Schedule"])[0]:
    #     schedule_tab()

if __name__ == "__main__":
    main()
