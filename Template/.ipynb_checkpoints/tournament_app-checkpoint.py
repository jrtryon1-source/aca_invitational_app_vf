import streamlit as st
import pandas as pd
import numpy as np
import os
import random
from pathlib import Path

# ==========================
# Page config
# ==========================
st.set_page_config(
    page_title="Speech & Debate Tournament Manager",
    layout="wide"
)

# ==========================
# Data folder
# ==========================
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

STUDENTS_FILE = DATA_DIR / "students.csv"
JUDGES_FILE = DATA_DIR / "judges.csv"
ROOMS_FILE = DATA_DIR / "rooms.csv"

# canonical round columns (consistent across the app)
ROUND_COLS = [
    "ROUND","AFF","NEG","JUDGE","ROOM",
    "WINNER","AFF SPEAKER POINTS","NEG SPEAKER POINTS"
]

# ==========================
# Utility function to save CSV
# ==========================
def save_csv(df, path):
    if isinstance(path, Path):
        path = str(path)
    df.to_csv(path, index=False)

# ==========================
# Callback functions for deletion
# ==========================
def delete_student(idx):
    st.session_state["students"] = st.session_state["students"].drop(idx).reset_index(drop=True)
    save_csv(st.session_state["students"], STUDENTS_FILE)

def delete_judge(idx):
    st.session_state["judges"] = st.session_state["judges"].drop(idx).reset_index(drop=True)
    save_csv(st.session_state["judges"], JUDGES_FILE)

# ==========================
# Load or initialize data
# ==========================
if "students" not in st.session_state:
    if STUDENTS_FILE.exists():
        st.session_state["students"] = pd.read_csv(STUDENTS_FILE)
    else:
        st.session_state["students"] = pd.DataFrame(columns=[
            "first_name","last_name","full_name","school","Wins","AverageSpeaks"
        ])

if "judges" not in st.session_state:
    if JUDGES_FILE.exists():
        st.session_state["judges"] = pd.read_csv(JUDGES_FILE)
    else:
        st.session_state["judges"] = pd.DataFrame(columns=[
            "first_name","last_name","full_name","CannotJudge"
        ])

if "rooms" not in st.session_state:
    if ROOMS_FILE.exists():
        st.session_state["rooms"] = pd.read_csv(ROOMS_FILE)
    else:
        st.session_state["rooms"] = pd.DataFrame(columns=["Room","Rounds"])

# rounds initialization (use canonical columns)
if "rounds" not in st.session_state:
    st.session_state["rounds"] = {}
    for i in range(1,5):
        st.session_state["rounds"][f"Round {i}"] = pd.DataFrame(columns=ROUND_COLS)

if "outrounds" not in st.session_state:
    st.session_state["outrounds"] = pd.DataFrame(columns=ROUND_COLS)

# ==========================
# Page Layout
# ==========================
st.title("Speech & Debate Tournament Manager")
tabs = st.tabs([
    "Tournament Central", "Judges", "Rooms",
    "Round 1", "Round 2", "Round 3", "Round 4", "Outrounds"
])

# ------------------------------------------------
# 🔢 Helper: Update Wins + AverageSpeaks in students
# ------------------------------------------------
def update_student_records():
    students_df = st.session_state["students"].copy()
    if students_df.empty:
        return

    # initialize counters
    students_df["Wins"] = 0
    total_points = {s: 0.0 for s in students_df["full_name"]}
    count_points = {s: 0 for s in students_df["full_name"]}

    # iterate through all rounds
    for r_name, df in st.session_state["rounds"].items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            aff = r.get("AFF", "")
            neg = r.get("NEG", "")
            winner = r.get("WINNER", "")
            # speaker points may be missing or NaN
            aff_pts = r.get("AFF SPEAKER POINTS", None)
            neg_pts = r.get("NEG SPEAKER POINTS", None)

            # wins logic (BYE handled here too)
            if winner == "AFF":
                if aff != "BYE":
                    students_df.loc[students_df["full_name"] == aff, "Wins"] += 1
                elif neg == "BYE" and aff != "BYE":
                    # aff vs BYE, AFF win
                    students_df.loc[students_df["full_name"] == aff, "Wins"] += 1
            elif winner == "NEG":
                if neg != "BYE":
                    students_df.loc[students_df["full_name"] == neg, "Wins"] += 1
                elif aff == "BYE" and neg != "BYE":
                    students_df.loc[students_df["full_name"] == neg, "Wins"] += 1
            # DBL LOSS → no wins

            # accumulate speaks if present and not BYE
            try:
                if aff and aff != "BYE" and aff_pts is not None and not (pd.isna(aff_pts)):
                    total_points[aff] += float(aff_pts)
                    count_points[aff] += 1
            except Exception:
                pass
            try:
                if neg and neg != "BYE" and neg_pts is not None and not (pd.isna(neg_pts)):
                    total_points[neg] += float(neg_pts)
                    count_points[neg] += 1
            except Exception:
                pass

    # compute averages
    average_list = []
    for s in students_df["full_name"]:
        if count_points.get(s, 0) > 0:
            avg = round(total_points[s] / count_points[s], 2)
        else:
            avg = 0.0
        average_list.append(avg)
    students_df["AverageSpeaks"] = average_list

    # sort by Wins desc, then AverageSpeaks desc
    students_df = students_df.sort_values(by=["Wins", "AverageSpeaks"], ascending=[False, False]).reset_index(drop=True)

    st.session_state["students"] = students_df
    save_csv(students_df, STUDENTS_FILE)

# -----------------------------------
# 🧰 Helper: Unified Editable Table UI
# -----------------------------------
def edit_round_table(round_name, judges_df, rooms_df):
    st.subheader(f"Edit {round_name}")

    # Load round DataFrame
    if "rounds" not in st.session_state or round_name not in st.session_state["rounds"]:
        st.warning(f"{round_name} not generated yet.")
        return

    df = st.session_state["rounds"][round_name]

    # Auto-randomize missing fields
    for idx, row in df.iterrows():
        if pd.isna(row.get("AFF SPEAKER POINTS")) or row["AFF SPEAKER POINTS"] == "":
            df.at[idx, "AFF SPEAKER POINTS"] = random.randint(25, 30)
        if pd.isna(row.get("NEG SPEAKER POINTS")) or row["NEG SPEAKER POINTS"] == "":
            df.at[idx, "NEG SPEAKER POINTS"] = random.randint(25, 30)
        if pd.isna(row.get("WINNER")) or row["WINNER"] == "":
            df.at[idx, "WINNER"] = random.choice(["AFF", "NEG"])

    st.session_state["rounds"][round_name] = df.copy()

    edited_rows = []

    # Build editable table
    headers = ["Round", "AFF", "NEG", "Judge", "Room", "Winner", "AFF SPEAKER POINTS", "NEG SPEAKER POINTS"]
    st.write(f"**{round_name} Matchups**")
    header_cols = st.columns([1.5, 2, 2, 2, 2, 1.5, 2, 2])
    for col, h in zip(header_cols, headers):
        col.write(f"**{h}**")

    for idx, row in df.iterrows():
        cols = st.columns([1.5, 2, 2, 2, 2, 1.5, 2, 2])

        # Editable columns
        round_label = cols[0].text_input("Round", value=row["ROUND"], key=f"{round_name}_round_{idx}")
        aff = cols[1].text_input("Aff", value=row["AFF"], key=f"{round_name}_aff_{idx}")
        neg = cols[2].text_input("Neg", value=row["NEG"], key=f"{round_name}_neg_{idx}")
        judge = cols[3].selectbox(
            "Judge",
            [""] + judges_df["Judge"].tolist(),
            index=([""] + judges_df["Judge"].tolist()).index(row["JUDGE"]) if row["JUDGE"] in judges_df["Judge"].tolist() else 0,
            key=f"{round_name}_judge_{idx}"
        )
        room = cols[4].selectbox(
            "Room",
            [""] + rooms_df["Room"].tolist(),
            index=([""] + rooms_df["Room"].tolist()).index(row["ROOM"]) if row["ROOM"] in rooms_df["Room"].tolist() else 0,
            key=f"{round_name}_room_{idx}"
        )

        # Winner auto-randomized but editable
        winner = cols[5].selectbox(
            "Winner",
            ["", "AFF", "NEG", "DBL LOSS"],
            index=(["", "AFF", "NEG", "DBL LOSS"].index(row["WINNER"])
                   if row["WINNER"] in ["", "AFF", "NEG", "DBL LOSS"] else 0),
            key=f"{round_name}_winner_{idx}"
        )

        # Editable speaker points (start with random 25–30)
        aff_speaks = cols[6].text_input("Aff Speaks", value=str(row["AFF SPEAKER POINTS"]), key=f"{round_name}_affspeaks_{idx}")
        neg_speaks = cols[7].text_input("Neg Speaks", value=str(row["NEG SPEAKER POINTS"]), key=f"{round_name}_negspeaks_{idx}")

        try: aff_speaks = float(aff_speaks)
        except: aff_speaks = 0
        try: neg_speaks = float(neg_speaks)
        except: neg_speaks = 0

        edited_rows.append([
            round_label, aff, neg, judge, room, winner, aff_speaks, neg_speaks
        ])

    # Save updates
    new_df = pd.DataFrame(edited_rows, columns=df.columns)
    st.session_state["rounds"][round_name] = new_df
    save_csv(new_df, f"{round_name.replace(' ', '_')}.csv")

    st.success(f"{round_name} updated successfully!")
    st.dataframe(new_df)

# ----------------------------------------
# Round generation helpers (consistent columns)
# ----------------------------------------
def generate_random_pairings_basic(round_number):
    """Simple random pairings generator that prevents rematches and tries to avoid duplicate BYEs."""
    round_name = f"Round {round_number}"

    students_df = st.session_state["students"]
    if students_df.empty:
        st.warning("No students available to generate pairings.")
        return

    # collect previous matchups and byed students
    prev_pairs = set()
    byed = set()
    for rn, df in st.session_state["rounds"].items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            a = r.get("AFF","")
            b = r.get("NEG","")
            if a and b:
                prev_pairs.add(tuple(sorted([a,b])))
            # record who had byes
            if b == "BYE" and a != "BYE":
                byed.add(a)
            if a == "BYE" and b != "BYE":
                byed.add(b)

    names = students_df["full_name"].tolist()
    random.shuffle(names)

    bye = None
    if len(names) % 2 == 1:
        candidates = [n for n in names if n not in byed]
        if not candidates:
            candidates = names
        bye = random.choice(candidates)
        names.remove(bye)

    matchups = []
    i = 0
    while i < len(names) - 1:
        s1 = names[i]
        s2 = names[i+1]
        # avoid simple rematch: if rematch, try to swap with later
        if tuple(sorted([s1,s2])) in prev_pairs:
            swapped = False
            for j in range(i+2, len(names)):
                if tuple(sorted([s1, names[j]])) not in prev_pairs:
                    names[i+1], names[j] = names[j], names[i+1]
                    s2 = names[i+1]
                    swapped = True
                    break
            if not swapped:
                # give up on this pair, skip forward to try again
                i += 1
                continue
        # random side assignment
        if random.choice([True, False]):
            aff, neg = s1, s2
        else:
            aff, neg = s2, s1

        # assign judge and room later in edit table; store blanks now
        matchups.append([round_name, aff, neg, "", "", "", "", ""])
        i += 2

    if bye:
        # choose side for bye randomly but try to alternate by round parity for balance
        if round_number % 2 == 1:
            aff, neg = bye, "BYE"
        else:
            aff, neg = "BYE", bye
        # default winner and speaks: winner gets average speaks later; set winner as AFF/NEG accordingly
        winner = "AFF" if aff != "BYE" else "NEG"
        aff_sp = students_df["AverageSpeaks"].mean() if (aff != "BYE" and not students_df["AverageSpeaks"].empty) else 0
        neg_sp = 0.0
        matchups.append([round_name, aff, neg, "", "", winner, aff_sp, neg_sp])

    df_new = pd.DataFrame(matchups, columns=ROUND_COLS)
    st.session_state["rounds"][round_name] = df_new
    save_csv(df_new, DATA_DIR / f"{round_name.replace(' ', '_')}.csv")
    # update students immediately
    update_student_records()
    st.success(f"Generated {len(df_new)} matchups for {round_name}")

# ----------------------------------------
# Rounds 1–2 UI
# ----------------------------------------
def round_tab_1_2(round_number):
    round_name = f"Round {round_number}"

    st.header(f"{round_name} - Pairings")

    # Initialize if missing
    if "rounds" not in st.session_state:
        st.session_state["rounds"] = {}

    students_df = st.session_state.get("students", pd.DataFrame())
    judges_df = st.session_state.get("judges", pd.DataFrame())
    rooms_df = st.session_state.get("rooms", pd.DataFrame())

    if students_df.empty:
        st.warning("Please add students first.")
        return

    # Randomize students & sides for first two rounds
    shuffled = students_df.sample(frac=1).reset_index(drop=True)
    matchups = []
    bye_assigned = False

    for i in range(0, len(shuffled), 2):
        if i + 1 < len(shuffled):
            aff, neg = shuffled.iloc[i]["full_name"], shuffled.iloc[i + 1]["full_name"]
            matchups.append([round_name, aff, neg, "", "", "", "", ""])
        else:
            # Handle bye
            bye = shuffled.iloc[i]["full_name"]
            matchups.append([round_name, bye, "BYE", "", "", "AFF", random.randint(25, 30), 0])
            bye_assigned = True

    df = pd.DataFrame(matchups, columns=["ROUND", "AFF", "NEG", "JUDGE", "ROOM", "WINNER", "AFF SPEAKER POINTS", "NEG SPEAKER POINTS"])
    st.session_state["rounds"][round_name] = df
    save_csv(df, f"{round_name.replace(' ', '_')}.csv")

    edit_round_table(round_name, judges_df, rooms_df)

# ----------------------------------------
# Rounds 3–4 UI (currently same generation; can be replaced with power-pairing)
# ----------------------------------------
def round_tab_3_4(round_number):
    round_name = f"Round {round_number}"

    st.header(f"{round_name} - Pairings")

    students_df = st.session_state.get("students", pd.DataFrame())
    judges_df = st.session_state.get("judges", pd.DataFrame())
    rooms_df = st.session_state.get("rooms", pd.DataFrame())

    if students_df.empty:
        st.warning("Please add students first.")
        return

    # Compute records & average speaks
    wins = {s["full_name"]: 0 for _, s in students_df.iterrows()}
    speaks = {s["full_name"]: [] for _, s in students_df.iterrows()}

    for rn, df in st.session_state.get("rounds", {}).items():
        if df.empty: continue
        for _, row in df.iterrows():
            if row["WINNER"] == "AFF":
                wins[row["AFF"]] += 1
            elif row["WINNER"] == "NEG":
                wins[row["NEG"]] += 1
            if row["AFF SPEAKER POINTS"]: speaks[row["AFF"]].append(float(row["AFF SPEAKER POINTS"]))
            if row["NEG SPEAKER POINTS"]: speaks[row["NEG"]].append(float(row["NEG SPEAKER POINTS"]))

    avg_speaks = {k: (sum(v) / len(v) if v else 0) for k, v in speaks.items()}
    students_df["Wins"] = students_df["full_name"].map(wins)
    students_df["AvgSpeaks"] = students_df["full_name"].map(avg_speaks)

    # Sort within record group
    grouped = {w: g.sort_values("AvgSpeaks", ascending=False) for w, g in students_df.groupby("Wins")}
    records = sorted(grouped.keys(), reverse=True)

    matchups = []
    bye_assigned = False

    for rec in records:
        pool = grouped[rec].copy()
        while len(pool) > 1:
            aff = pool.iloc[0]["full_name"]
            neg = pool.iloc[-1]["full_name"]
            pool = pool.iloc[1:-1]
            matchups.append([round_name, aff, neg, "", "", "", "", ""])
        if len(pool) == 1:
            # Bye for leftover
            bye = pool.iloc[0]["full_name"]
            matchups.append([round_name, bye, "BYE", "", "", "AFF", random.randint(25, 30), 0])
            bye_assigned = True

    df = pd.DataFrame(matchups, columns=["ROUND", "AFF", "NEG", "JUDGE", "ROOM", "WINNER", "AFF SPEAKER POINTS", "NEG SPEAKER POINTS"])
    st.session_state["rounds"][round_name] = df
    save_csv(df, f"{round_name.replace(' ', '_')}.csv")

    edit_round_table(round_name, judges_df, rooms_df)

# ==========================
# 1️⃣ Tournament Central - Students
# ==========================
with tabs[0]:
    st.header("Tournament Central - Students")

    # --- Upload Students CSV ---
    uploaded_file = st.file_uploader(
        "Upload Students CSV (first_name,last_name,school)",
        type=["csv"],
        key="students_upload"
    )
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)
        required_cols = {"first_name", "last_name", "school"}
        if not required_cols.issubset(df_upload.columns):
            st.error(f"CSV must include columns: {required_cols}")
        else:
            df_upload["full_name"] = df_upload["first_name"].str.strip() + " " + df_upload["last_name"].str.strip()
            for col in ["Wins", "AverageSpeaks"]:
                if col not in df_upload.columns:
                    df_upload[col] = 0
            st.session_state["students"] = df_upload[["first_name","last_name","full_name","school","Wins","AverageSpeaks"]]
            save_csv(st.session_state["students"], STUDENTS_FILE)
            st.success(f"Loaded {len(df_upload)} students")
            # ensure records are consistent
            update_student_records()

    # --- Add Student Form ---
    with st.form("add_student", clear_on_submit=True):
        first_name = st.text_input("First Name", key="student_first")
        last_name = st.text_input("Last Name", key="student_last")
        school = st.text_input("School", key="student_school")
        submitted = st.form_submit_button("Add Student")

        if submitted:
            if not first_name.strip() or not last_name.strip() or not school.strip():
                st.error("Please provide First Name, Last Name, and School")
            else:
                full_name = f"{first_name.strip()} {last_name.strip()}"
                new_row = {
                    "first_name": first_name.strip(),
                    "last_name": last_name.strip(),
                    "full_name": full_name,
                    "school": school.strip(),
                    "Wins": 0,
                    "AverageSpeaks": 0
                }
                st.session_state["students"] = pd.concat(
                    [st.session_state["students"], pd.DataFrame([new_row])],
                    ignore_index=True
                )
                save_csv(st.session_state["students"], STUDENTS_FILE)
                update_student_records()
                st.success(f"Added {full_name}")

    # Render Students Table (sorted by wins/avg speaks already via update_student_records)
    students_df = st.session_state["students"]
    st.subheader("All Students")
    if students_df.empty:
        st.info("No students yet.")
    else:
        header_cols = st.columns([2,2,2,1,1,1])
        headers = ["First Name","Last Name","School","Wins","AverageSpeaks","Delete"]
        for col, title in zip(header_cols, headers):
            col.write(f"**{title}**")

        for i, row in students_df.iterrows():
            cols = st.columns([2,2,2,1,1,1])
            cols[0].write(row["first_name"])
            cols[1].write(row["last_name"])
            cols[2].write(row["school"])
            cols[3].write(row["Wins"])
            cols[4].write(row["AverageSpeaks"])

            if cols[5].button("Delete", key=f"del_student_{i}"):
                if st.confirm(f"Are you sure you want to delete **{row['full_name']}**?"):
                    delete_student(i)
                    st.experimental_rerun()

# ==========================
# 2️⃣ Judges Tab
# ==========================
with tabs[1]:
    st.header("Judges")

    # --- Add Judge Form ---
    with st.form("add_judge", clear_on_submit=True):
        first_name = st.text_input("First Name", key="judge_first")
        last_name = st.text_input("Last Name", key="judge_last")
        cannot_judge = st.multiselect(
            "Cannot Judge (select students from list)",
            options=st.session_state["students"]["full_name"].tolist() if not st.session_state["students"].empty else []
        )
        submitted = st.form_submit_button("Add Judge")
        if submitted and first_name and last_name:
            full_name = f"{first_name.strip()} {last_name.strip()}"
            cannot_judge_str = ",".join(cannot_judge)
            st.session_state["judges"].loc[len(st.session_state["judges"])] = [
                first_name, last_name, full_name, cannot_judge_str
            ]
            save_csv(st.session_state["judges"], JUDGES_FILE)
            st.success(f"Added Judge: {full_name}")

    # --- Render Judges Table with Delete Buttons ---
    st.subheader("All Judges")
    df_judges = st.session_state["judges"]
    if df_judges.empty:
        st.info("No judges yet.")
    else:
        header_cols = st.columns([3,3,3,1])
        header_cols[0].write("First Name")
        header_cols[1].write("Last Name")
        header_cols[2].write("Cannot Judge")
        header_cols[3].write("Delete")

        for i, row in df_judges.iterrows():
            cols = st.columns([3,3,3,1])
            cols[0].write(row["first_name"])
            cols[1].write(row["last_name"])
            cols[2].write(row["CannotJudge"])
            cols[3].button("Delete", key=f"del_judge_{i}", on_click=delete_judge, args=(i,))

# ==========================
# 3️⃣ Rooms
# ==========================
with tabs[2]:
    st.header("Rooms")
    st.subheader("Add Room")
    with st.form("add_room"):
        room_name = st.text_input("Room Name")
        submitted = st.form_submit_button("Add Room")
        if submitted and room_name:
            st.session_state["rooms"].loc[len(st.session_state["rooms"])] = [room_name, ""]
            save_csv(st.session_state["rooms"], ROOMS_FILE)
            st.success(f"Added room {room_name}")
    st.subheader("All Rooms")
    st.dataframe(st.session_state["rooms"], use_container_width=True)

# ==========================
# 4️⃣–7️⃣ Rounds
# ==========================
with tabs[3]:
    round_tab_1_2(1)
with tabs[4]:
    round_tab_1_2(2)
with tabs[5]:
    round_tab_3_4(3)
with tabs[6]:
    round_tab_3_4(4)

# ==========================
# 8️⃣ Outrounds
# ==========================
with tabs[7]:
    st.header("Outrounds (Top 4 Debaters)")
    st.write("Create outround bracket after prelim rounds.")
    top4 = st.session_state["students"].sort_values(by=["Wins","AverageSpeaks"], ascending=[False, False]).head(4)
    st.dataframe(top4[["full_name","Wins","AverageSpeaks"]], use_container_width=True)

