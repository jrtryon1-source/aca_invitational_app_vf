import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import time
import random
import ast



# ==========================
# Page config
# ==========================
st.set_page_config(page_title="Tournament Hub", layout="wide")

# ==========================
# Data folder
# ==========================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
STUDENTS_FILE = Path(__file__).parent / "student_list" / "students.csv"
ROOMS_FILE = Path(__file__).parent / "student_list" / "rooms.csv"

# ==========================
# Utility: Load or init Tables
# ==========================
def load_students():
    if "students" not in st.session_state:
        if STUDENTS_FILE.exists():
            st.session_state["students"] = pd.read_csv(STUDENTS_FILE)
        else:
            st.session_state["students"] = pd.DataFrame(columns=[
                "First Name", "Last Name", "Full Name", "School", "Wins", "Losses", "Average Speaker Points"
            ])

    print(st.session_state["students"])
    return st.session_state["students"]

def save_students():
    st.session_state["students"].to_csv(STUDENTS_FILE, index=False)

def save_rooms():
    st.session_state["students"].to_csv(ROOMS_FILE, index=False)

def load_judges():
    if "judges" not in st.session_state:
        st.session_state["judges"] = pd.DataFrame(columns=[
            "First Name", "Last Name", "Full Name", "Cannot Judge"
            ])

def load_rooms():
    if "rooms" not in st.session_state:
        st.session_state["rooms"] = pd.DataFrame(columns=[
            "Room Number", "Rounds Available"
            ])
        
def load_schedule():
    if "schedule" not in st.session_state:
        st.session_state["schedule"] = pd.DataFrame({
            "Round": ["Round 1 🏛️", "Round 2 🏛️", "Round 3 🏛️", "Round 4 🏛️", "Semifinals 🏆", "Final 🏆"],
            "Time": ["09:00 AM", "10:30 AM", "12:00 PM", "01:30 PM", "03:00 PM", "04:30 PM"]
        })
    return st.session_state["schedule"]

def load_rounds():
    if "rounds" not in st.session_state:
        st.session_state["rounds"] = pd.DataFrame({
            "Round_ID", "Round", "Student", "Opponent", "Side", "Win", "Speaker Points", "Room", "Judge"
        })
    return st.session_state["round"]

def assign_judge(aff, neg):
    if "judges" not in st.session_state or st.session_state["judges"].empty:
        return ""
    
    eligible_judges = []
    for _, judge_row in st.session_state["judges"].iterrows():
        cannot_judge = judge_row.get("Cannot Judge", [])
        # Ensure cannot_judge is a list
        if isinstance(cannot_judge, str):
            cannot_judge = cannot_judge.split(",")  # or parse your format
        # Check if this judge can judge both students
        if aff not in cannot_judge and neg not in cannot_judge:
            eligible_judges.append(judge_row["Full Name"])
    
    return random.choice(eligible_judges) if eligible_judges else ""

def prepare_aff_neg_table(round_df):
    """
    Converts the generated round_df (Student/Opponent/Side) into a display table
    with AFF/NEG columns for easy editing in Streamlit.
    """
    aff_neg_rows = []
    for _, row in round_df.iterrows():
        if row["Side"] == "AFF":
            aff = row["Student"]
            neg = row["Opponent"]
        else:
            aff = row["Opponent"]
            neg = row["Student"]

        aff_neg_rows.append({
            "Round_ID": row["Round_ID"],
            "Round": row["Round"],
            "AFF": aff,
            "NEG": neg,
            "JUDGE": "",
            "ROOM": "",
            "WIN": "",
            "AFF SPEAKS": "",
            "NEG SPEAKS": ""
        })

    return pd.DataFrame(aff_neg_rows)

def update_student_records_from_rounds():
    """Recalculate wins, losses, and average speaks from all rounds."""
    if "students" not in st.session_state or st.session_state["students"].empty:
        return

    students_df = st.session_state["students"].copy()
    students_df["Wins"] = 0
    students_df["Losses"] = 0
    students_df["Average Speaker Points"] = 0.0

    if "rounds" not in st.session_state or st.session_state["rounds"].empty:
        st.session_state["students"] = students_df
        return

    speaks_totals = {s: [] for s in students_df["Full Name"]}

    for _, r in st.session_state["rounds"].iterrows():
        student = r["Student"]
        if student not in speaks_totals:
            continue

        # Speaker points
        try:
            sp = float(r["Speaker Points"])
        except (ValueError, TypeError):
            sp = 0.0
        speaks_totals[student].append(sp)

        # Wins & losses
        if r["Win"] == "AFF" and r["Side"] == "AFF":
            students_df.loc[students_df["Full Name"] == student, "Wins"] += 1
        elif r["Win"] == "NEG" and r["Side"] == "NEG":
            students_df.loc[students_df["Full Name"] == student, "Wins"] += 1
        elif r["Win"] in ["AFF", "NEG"] and r["Side"] != r["Win"]:
            students_df.loc[students_df["Full Name"] == student, "Losses"] += 1
        elif r["Win"] == "DOUBLE LOSS":
            students_df.loc[students_df["Full Name"] == student, "Losses"] += 1

    # Calculate averages
    for s, speaks in speaks_totals.items():
        avg = round(sum(speaks) / len(speaks), 2) if speaks else 0.0
        students_df.loc[students_df["Full Name"] == s, "Average Speaker Points"] = avg

    # Sort by Wins then AverageSpeaks
    students_df = students_df.sort_values(by=["Wins", "Average Speaker Points"], ascending=[False, False])
    st.session_state["students"] = students_df

    # Save to file (optional)
    STUDENTS_FILE = Path(__file__).parent / "student_list" / "students.csv"
    students_df.to_csv(STUDENTS_FILE, index=False)

# ==========================
# UI Elements
# ==========================
def tournament_hub():
    st.title("🏆 Tournament Hub")

    # 🔄 Always refresh student standings from rounds
    if "rounds" in st.session_state and not st.session_state["rounds"].empty:
        update_student_records_from_rounds()

    students_df = load_students()

    # --------------------------
    # Add a student form
    # --------------------------
    with st.form("add_student", clear_on_submit=True):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        school = st.text_input("School")
        submitted = st.form_submit_button("Add Student")

        if submitted:
            if not first_name or not last_name:
                st.error("Please enter both first and last name")
            else:
                full_name = f"{first_name.strip()} {last_name.strip()}"
                new_row = {
                    "First Name": first_name.strip(),
                    "Last Name": last_name.strip(),
                    "Full Name": full_name,
                    "School": school,
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

    # --------------------------
    # Display all students
    # --------------------------
    st.subheader("📊 Current Standings")

    students_df = st.session_state["students"].copy()
    if students_df.empty:
        st.info("No students added yet.")
    else:
        # Sort by Wins descending, then AverageSpeaks descending
        students_df = students_df.sort_values(
            by=["Wins", "Average Speaker Points"],
            ascending=[False, False]
        )
        st.dataframe(students_df, use_container_width=True)

    # ==========================
    # Remove a student
    # ==========================
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

def judges_tab():
    st.header("⚖️ Judges Pool")

    students_df = st.session_state.get("students", pd.DataFrame())
    student_names = students_df["Full Name"].tolist() if not students_df.empty else []

    judges_df = load_judges()

    # --------------------------
    # Add Judge Form
    # --------------------------
    with st.form("add_judge", clear_on_submit=True):
        first_name = st.text_input("First Name")
        last_name = st.text_input("Last Name")
        cannot_judge = st.multiselect(
            "Cannot Judge (Select students you cannot judge)",
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

    # --------------------------
    # Display all judges
    # --------------------------
    st.subheader("All Judges")
    judges_df = st.session_state["judges"]
    if judges_df.empty:
        st.info("No judges added yet.")
    else:
        st.dataframe(judges_df, use_container_width=True)

    # ==========================
    # Remove a judge
    # ==========================
    st.subheader("Remove a Judge")

    judges_df = st.session_state["judges"]

    if not judges_df.empty:
        judge_to_remove = st.selectbox(
            "Select a student to remove",
            options=[""] + judges_df["Full Name"].tolist(),
            index=0,
        )   

        if st.button("Remove Judge"):
            if judge_to_remove:
                st.session_state["judges"] = judges_df[judges_df["Full Name"] != judge_to_remove]
                st.session_state["judges"].reset_index(drop=True, inplace=True)
                st.success(f"Removed {judge_to_remove}")
            else:
                st.warning("Please select a judge to remove.")
    else:
        st.info("No judges available to remove.")

def rooms_tab():
    st.header("🏛️ Available Rooms")

    rooms_df = load_rooms()

    # Add a room
    with st.form("add_room", clear_on_submit=True):
        room_num = st.text_input("Room Number")
        round_available = st.multiselect(
            "Available Rounds (Select rounds this room is available for)",
            options=('All', '1', '2', '3', '4', 'Semifinals', 'Finals')
        )
        submitted = st.form_submit_button("Add Room")
        
        if submitted:
            if not room_num or not round_available:
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
    # Display Rooms
    # --------------------------
    # Display all students
    # --------------------------
    
    st.subheader("All Rooms")
    rooms_df = st.session_state["rooms"].copy()

    if rooms_df.empty:
        st.info("No rooms added yet.")
    else:
        st.dataframe(rooms_df, use_container_width=True)
    
    # Remove a room
    st.subheader("Remove a room")

    rooms_df = st.session_state["rooms"]

    if not rooms_df.empty:
        rooms_to_remove = st.selectbox(
            "Select a room to remove",
            options=[""] + rooms_df["Room Number"].tolist(),
            index=0,
        )   

        if st.button("Remove room"):
            if rooms_to_remove:
                st.session_state["rooms"] = rooms_df[rooms_df["Room Number"] != rooms_to_remove]
                st.session_state["rooms"].reset_index(drop=True, inplace=True)
                st.success(f"Removed {rooms_to_remove}")
            else:
                st.warning("Please select a room to remove.")
    else:
        st.info("No rooms available to remove.")

def schedule_tab():
    st.header("🕰️ Tournament Schedule")
    st.markdown("<p style='font-size:14px; font-weight:normal; margin-bottom:5px;'>Update scheduling to display on round postings </p>", unsafe_allow_html=True)

    schedule_df = load_schedule()

    for i, row in schedule_df.iterrows():
        # Center the columns using an empty column trick
        col_left, col_center, col_right = st.columns([1, 2, 1])
        with col_center:
            inner_col1, inner_col2 = st.columns([1, 1])  # Round name vs time input close together
            with inner_col1:
                st.write(f"**{row['Round']}**")
            with inner_col2:
                # Parse current time
                time_str = row["Time"].strip()
                if not time_str:
                    default_time = time(9,0)
                else:
                    t_part, ampm = time_str[:-3], time_str[-2:]
                    h_str, m_str = t_part.split(":")
                    h, m = int(h_str), int(m_str)
                    if ampm.upper() == "PM" and h != 12:
                        h += 12
                    elif ampm.upper() == "AM" and h == 12:
                        h = 0
                    default_time = time(h, m)

                # Shrink width by using columns trick again
                st.session_state["schedule"].at[i, "Time"] = st.time_input(
                    label="",
                    value=default_time,
                    key=f"round_time_{i}",
                    label_visibility="collapsed"
                ).strftime("%I:%M %p")


    st.subheader("📋 Current Schedule")
    st.dataframe(st.session_state["schedule"], use_container_width=True)

def generate_matchups(round_number: int):
    """
    Generates matchups for a given round according to your rules:
    1. Students aff twice, neg twice across 4 rounds.
    2. Byes handled automatically with 25 speaker points.
    3. Rounds 1-2: ensure each student aff once, neg once.
    4. Rounds 3-4: power-pair by Wins, then Average Speaker Points.
    5. No repeat matches.
    
    Returns a DataFrame:
    Round_ID | Round | Student | Opponent | Side | Win | Speaker Points | Room | Judge
    """
    students_df = st.session_state["students"].copy()
    rounds_df = st.session_state.get("rounds", pd.DataFrame())
    
    # Track all previous matchups to avoid repeats
    previous_matches = set()
    if not rounds_df.empty:
        for _, r in rounds_df.iterrows():
            s1, s2 = r["Student"], r["Opponent"]
            previous_matches.add(tuple(sorted([s1, s2])))

    # Determine byes
    names = students_df["Full Name"].tolist()
    bye = None
    if len(names) % 2 == 1:
        # Choose student with fewest byes so far
        bye_counts = students_df.get("Byes", pd.Series(0, index=students_df.index))
        min_byes = bye_counts.min()
        candidates = students_df[bye_counts == min_byes]["Full Name"].tolist()
        bye = random.choice(candidates)
        names.remove(bye)
    
    # Round 1-2: simple alternating sides
    matchups = []
    if round_number in [1,2]:
        random.shuffle(names)
        for i in range(0, len(names), 2):
            s1, s2 = names[i], names[i+1]
            aff, neg = (s1, s2) if round_number % 2 == 1 else (s2, s1)
            matchups.append({
                "Round_ID": f"{round_number}.{i//2+1}",
                "Round": round_number,
                "Student": aff,
                "Opponent": neg,
                "Side": "AFF",
                "Win": "",
                "Speaker Points": "",
                "Room": "",
                "Judge": ""
            })
        if bye:
            matchups.append({
                "Round_ID": f"{round_number}.{len(names)//2+1}",
                "Round": round_number,
                "Student": bye,
                "Opponent": "BYE",
                "Side": "AFF" if round_number % 2 == 1 else "NEG",
                "Win": "AFF",
                "Speaker Points": 25,
                "Room": "",
                "Judge": ""
            })
    
    # Round 3-4: power-paired by Wins + Avg Speaker Points
    else:
        # Sort students by Wins descending, then Average Speaker Points descending
        standings = students_df.sort_values(by=["Wins","Average Speaker Points"], ascending=[False, False]).copy()
        names_sorted = standings["Full Name"].tolist()
        
        # Pair within brackets
        used = set()
        i = 0
        while i < len(names_sorted)-1:
            s1, s2 = names_sorted[i], names_sorted[i+1]
            if tuple(sorted([s1, s2])) in previous_matches:
                # Skip to next if repeat
                i += 1
                continue
            side = "AFF" if i % 2 == 0 else "NEG"
            matchups.append({
                "Round_ID": f"{round_number}.{i//2+1}",
                "Round": round_number,
                "Student": s1,
                "Opponent": s2,
                "Side": side,
                "Win": "",
                "Speaker Points": "",
                "Room": "",
                "Judge": ""
            })
            used.update([s1, s2])
            i += 2
        
        # Handle bye if odd
        if bye and bye not in used:
            matchups.append({
                "Round_ID": f"{round_number}.{len(matchups)+1}",
                "Round": round_number,
                "Student": bye,
                "Opponent": "BYE",
                "Side": "AFF",
                "Win": "AFF",
                "Speaker Points": 25,
                "Room": "",
                "Judge": ""
            })
    
    df_round = pd.DataFrame(matchups)
    return df_round

import streamlit as st
import pandas as pd
import random
import ast

# ---------- helpers ----------
def _parse_cannot_judge(cell) -> set:
    """Return a cleaned set of names from a 'Cannot Judge' cell."""
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return set()
    s = str(cell).strip()
    if not s:
        return set()
    # Try list literal first: "['A', 'B']"
    try:
        val = ast.literal_eval(s)
        if isinstance(val, (list, tuple)):
            return set(x.strip() for x in val if str(x).strip())
    except Exception:
        pass
    # Fallback: CSV string "A,B"
    return set(x.strip() for x in s.split(",") if x.strip())

def _judge_can_cover(judges_df: pd.DataFrame, judge_name: str, aff: str, neg: str) -> bool:
    if judge_name == "" or judge_name is None:
        return False
    row = judges_df.loc[judges_df["Full Name"] == judge_name]
    if row.empty:
        return False
    cannot = _parse_cannot_judge(row.iloc[0].get("Cannot Judge", ""))
    return (aff not in cannot) and (neg not in cannot)

def _eligible_rooms_for_round(rooms_df: pd.DataFrame, round_number: int) -> list:
    """Return list of room names where Rounds Available contains round_number or 'All'."""
    if rooms_df is None or rooms_df.empty:
        return []

    eligible = []
    for _, r in rooms_df.iterrows():
        rv = r.get("Rounds Available", "All")

        # --- SAFER NAN CHECK ---
        try:
            if pd.isna(rv).item() if hasattr(pd.isna(rv), "item") else pd.isna(rv):
                rounds_available = {"All"}
            else:
                s = str(rv).strip()
                # Try literal eval, fallback to CSV parsing
                try:
                    parsed = ast.literal_eval(s)
                    if isinstance(parsed, (list, tuple, set)):
                        rounds_available = {str(x).strip() for x in parsed}
                    else:
                        rounds_available = {str(parsed).strip()}
                except Exception:
                    # Handle CSV-style or plain strings
                    if "," in s:
                        rounds_available = {x.strip() for x in s.split(",")}
                    else:
                        rounds_available = {s}
        except Exception:
            # Defensive fallback for weird datatypes
            rounds_available = {"All"}

        if "All" in rounds_available or str(round_number) in rounds_available:
            room_name = r.get("Room Number", "")
            if room_name:
                eligible.append(room_name)

    return eligible

def _assign_judges(display_round: pd.DataFrame, judges_df: pd.DataFrame) -> pd.DataFrame:
    """Greedy, one judge per debate, respecting Cannot Judge."""
    if judges_df is None or judges_df.empty:
        display_round["JUDGE"] = ""
        return display_round
    judge_pool = judges_df["Full Name"].dropna().tolist()
    used = set()
    for idx, row in display_round.iterrows():
        aff, neg = row["AFF"], row["NEG"]
        random.shuffle(judge_pool)
        assigned = ""
        for j in judge_pool:
            if j in used:
                continue
            if _judge_can_cover(judges_df, j, aff, neg):
                assigned = j
                used.add(j)
                break
        display_round.at[idx, "JUDGE"] = assigned
    return display_round

def _assign_rooms(display_round: pd.DataFrame, rooms_df: pd.DataFrame, round_number: int) -> pd.DataFrame:
    """One room per debate; respect Rounds Available; leave blank if not enough rooms."""
    rooms = _eligible_rooms_for_round(rooms_df, round_number) if rooms_df is not None else []
    random.shuffle(rooms)
    assigned = []
    for _ in range(len(display_round)):
        assigned.append(rooms.pop() if rooms else "")
    display_round["ROOM"] = assigned
    return display_round

def round_tab(
    round_number: int,
    generate_matchups=None,           # optional
    prepare_aff_neg_table=None        # optional
):
    st.header(f"✒️ Round {round_number}")

    # ---------- Fallback helpers ----------
    def _default_generate_matchups(rn: int) -> pd.DataFrame:
        students_df = st.session_state.get("students", pd.DataFrame(columns=["Full Name"]))
        names = students_df.get("Full Name", pd.Series(dtype=str)).dropna().tolist()
        random.shuffle(names)
        pairs = []
        # simple pairs + BYE if odd
        for i in range(0, len(names) - 1, 2):
            pairs.append({"Round": rn, "AFF": names[i], "NEG": names[i+1]})
        if len(names) % 2 == 1:
            pairs.append({"Round": rn, "AFF": names[-1], "NEG": "BYE"})
        return pd.DataFrame(pairs, columns=["Round", "AFF", "NEG"])

    def _default_prepare_aff_neg_table(raw_df: pd.DataFrame) -> pd.DataFrame:
        out = raw_df[["Round","AFF","NEG"]].copy() if {"AFF","NEG","Round"}.issubset(raw_df.columns) else pd.DataFrame(columns=["Round","AFF","NEG"])
        # add editor fields
        for c in ["JUDGE", "ROOM", "WIN", "AFF SPEAKS", "NEG SPEAKS"]:
            if c not in out.columns:
                out[c] = ""
        return out[["Round","AFF","NEG","JUDGE","ROOM","WIN","AFF SPEAKS","NEG SPEAKS"]]

    gm = generate_matchups or _default_generate_matchups
    prep = prepare_aff_neg_table or _default_prepare_aff_neg_table

    # ---------- keys ----------
    round_key = f"round_{round_number}"
    orig_key = f"{round_key}_orig"
    edited_key = f"{round_key}_edited"
    editor_widget_key = f"editor_widget_{round_number}"

    # ---------- initialize holder ----------
    if round_key not in st.session_state:
        st.session_state[round_key] = pd.DataFrame(columns=[
            "Round","AFF","NEG","JUDGE","ROOM","WIN","AFF SPEAKS","NEG SPEAKS"
        ])

    # ---------- GENERATE ----------
    if st.session_state[round_key].empty:
        if st.button(f"Generate Round {round_number}", key=f"gen_btn_{round_number}"):
            raw = gm(round_number)
            disp = prep(raw)

            judges_df = st.session_state.get("judges", pd.DataFrame(columns=["Full Name","Cannot Judge"]))
            rooms_df = st.session_state.get("rooms", pd.DataFrame(columns=["Room Number","Rounds Available"]))

            disp = _assign_judges(disp, judges_df)
            disp = _assign_rooms(disp, rooms_df, round_number)

            # init editable fields to string
            for c in ["WIN","AFF SPEAKS","NEG SPEAKS"]:
                disp[c] = disp[c].fillna("").astype(str)

            # set stable source for editor
            st.session_state[round_key] = disp.copy()
            st.session_state[orig_key] = disp.copy()
            st.session_state[edited_key] = disp.copy()

            st.rerun()
        else:
            st.info(f"Click **Generate Round {round_number}** to create matchups.")
            return

    # ---------- EDIT (stable pattern) ----------
    st.subheader("📋 Current Round Matchups")

    # Ensure orig exists (stable source for the editor)
    if orig_key not in st.session_state:
        st.session_state[orig_key] = st.session_state[round_key].copy()
    df_to_edit = st.session_state[orig_key].copy()

    # Cast editable columns to strings for stability
    for col in ["WIN","AFF SPEAKS","NEG SPEAKS"]:
        if col in df_to_edit:
            df_to_edit[col] = df_to_edit[col].fillna("").astype(str)

    column_config = {
        "WIN": st.column_config.SelectboxColumn(
            "WIN", options=["", "AFF", "NEG", "DOUBLE LOSS"]
        ),
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

    # Save the edited copy ONLY (do not replace editor's source mid-render)
    st.session_state[edited_key] = edited_df.copy()

    # ---------- FINALIZE ----------
    if st.button(f"✅ Finalize Round {round_number}", key=f"finalize_btn_{round_number}"):
        # Commit edited back to round table
        st.session_state[round_key] = st.session_state[edited_key].copy()
        st.session_state[orig_key] = st.session_state[edited_key].copy()

        # Append to global rounds table in side-based format
        if "rounds" not in st.session_state:
            st.session_state["rounds"] = pd.DataFrame(columns=[
                "Round", "Student", "Opponent", "Side", "Win", "Speaker Points", "Room", "Judge"
            ])

        to_append = []
        for _, r in st.session_state[round_key].iterrows():
            # AFF line
            to_append.append({
                "Round": r["Round"],
                "Student": r["AFF"],
                "Opponent": r["NEG"],
                "Side": "AFF",
                "Win": r["WIN"],
                "Speaker Points": r["AFF SPEAKS"],
                "Room": r["ROOM"],
                "Judge": r["JUDGE"]
            })
            # NEG line
            to_append.append({
                "Round": r["Round"],
                "Student": r["NEG"],
                "Opponent": r["AFF"],
                "Side": "NEG",
                "Win": r["WIN"],
                "Speaker Points": r["NEG SPEAKS"],
                "Room": r["ROOM"],
                "Judge": r["JUDGE"]
            })

        if to_append:
            st.session_state["rounds"] = pd.concat(
                [st.session_state["rounds"], pd.DataFrame(to_append)],
                ignore_index=True
            )

        # Update standings visible on Tournament Hub
        if "update_student_records_from_rounds" in globals():
            update_student_records_from_rounds()

        st.success(f"✅ Round {round_number} finalized. Standings updated on Tournament Hub.")

# ==========================
# Run Everything
# ==========================

def main():
    st.title("🏛️ ACA Classical Qualifier")

    #tabs = st.tabs(["Tournament Hub", "Judges Pool", "Rooms", "Schedule"])
    tabs = st.tabs(["Tournament Hub", "Judges Pool", "Rooms", "Round One", "Round Two", "Round Three", "Round Four"])

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
    with tabs[6]:
        round_tab(4)

    #print(generate_matchups(1))
    #print(generate_matchups(2))

    #with  tabs[3]:
    #    schedule_tab()

if __name__ == "__main__":
    main()
