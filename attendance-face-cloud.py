import os
import time
#from datetime import datetime
#from datetime import datetime, UTC
from datetime import datetime, UTC, timezone, timedelta

import io

import numpy as np
import cv2
from PIL import Image
import streamlit as st
from supabase import create_client, Client
import pandas as pd
from collections import defaultdict


# ---------------- Config ----------------
st.set_page_config(page_title="Face Attendance (Supabase)", layout="wide")

# Buckets and filenames
FACES_BUCKET = "faces"
MODELS_BUCKET = "models"
LBPH_MODEL_FILENAME = "lbph_model.xml"
LABEL_MAP_FILENAME = "label_to_empid.npy"
TMP_DIR = "/tmp"

# Haar cascade
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ---------------- Supabase clients ----------------
# Secrets (use Cloud Secrets or local .streamlit/secrets.toml)
SUPABASE_URL = st.secrets["SUPABASE_URL"].rstrip("/") + "/"  # enforce trailing slash
SUPABASE_KEY = st.secrets['SUPABASE_KEY']
SUPABASE_ANON_KEY = st.secrets["SUPABASE_ANON_KEY"]
SUPABASE_SERVICE_KEY = st.secrets["SUPABASE_SERVICE_KEY"]

# Public client (safe, uses anon key) for user-facing queries and auth
#supabase_public = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
supabase_public = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
# Admin client (privileged, uses service role key) for uploads, retraining, inserts
supabase_admin = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------- Utilities ----------------
def ensure_tmp_dir():
    os.makedirs(TMP_DIR, exist_ok=True)

def preprocess_face_from_pil(pil_img):
    """Detect, crop, and resize face to 200x200 grayscale. Returns (face_resized, rect, gray)."""
    rgb = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return None, None, gray
    # Select largest face
    (x, y, w, h) = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    roi = gray[y : y + h, x : x + w]
    face_resized = cv2.resize(roi, (200, 200))
    return face_resized, (x, y, w, h), gray

def register_employee_if_missing(empid, name):
    empid_norm = empid.upper().strip()
    existing = supabase_public.table("users").select("employee_id").eq("employee_id", empid_norm).execute()
    if not existing.data:
        supabase_admin.table("users").insert({"employee_id": empid_norm, "name": name}).execute()

def upload_face_sample_to_storage(empid_norm, filename, image_array_200x200):
    """Write to /tmp and upload to faces bucket under EMPID/filename."""
    ensure_tmp_dir()
    tmp_path = os.path.join(TMP_DIR, filename)
    cv2.imwrite(tmp_path, image_array_200x200)
    with open(tmp_path, "rb") as f:
        supabase_admin.storage.from_(FACES_BUCKET).upload(f"{empid_norm}/{filename}", f.read(), {"upsert": "true"})

def list_employee_folders():
    """List top-level entries in faces bucket (employee folders)."""
    return supabase_admin.storage.from_(FACES_BUCKET).list("")

def list_files_in_folder(folder):
    """List files inside a given folder in faces bucket."""
    return supabase_admin.storage.from_(FACES_BUCKET).list(folder)

def download_file_bytes(bucket, path):
    return supabase_admin.storage.from_(bucket).download(path)

def upload_model_to_storage(model_path, label_path):
    with open(model_path, "rb") as f:
        supabase_admin.storage.from_(MODELS_BUCKET).upload(LBPH_MODEL_FILENAME, f.read(), {"upsert": "true"})
    with open(label_path, "rb") as f:
        supabase_admin.storage.from_(MODELS_BUCKET).upload(LABEL_MAP_FILENAME, f.read(), {"upsert": "true"})

def load_model_from_storage():
    try:
        ensure_tmp_dir()
        # Download model
        model_bytes = supabase_admin.storage.from_(MODELS_BUCKET).download(LBPH_MODEL_FILENAME)
        tmp_model_path = os.path.join(TMP_DIR, LBPH_MODEL_FILENAME)
        with open(tmp_model_path, "wb") as f:
            f.write(model_bytes)
        # Download labels
        labels_bytes = supabase_admin.storage.from_(MODELS_BUCKET).download(LABEL_MAP_FILENAME)
        tmp_label_path = os.path.join(TMP_DIR, LABEL_MAP_FILENAME)
        with open(tmp_label_path, "wb") as f:
            f.write(labels_bytes)

        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(tmp_model_path)
        label_map = np.load(tmp_label_path, allow_pickle=True).item()
        return recognizer, label_map
    except Exception as e:
        st.error(f"Failed to load model from storage: {e}")
        return None, None

# Define Singapore timezone
SGT = timezone(timedelta(hours=8))

def format_sgt(ts_str):
    ts_utc = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    return ts_utc.astimezone(SGT)

def calculate_total_hours(empid, notes=[]):
    res = (
        supabase.table("attendance")
        .select("timestamp, method")
        .eq("employee_id", empid.strip().upper())
        .order("timestamp")  # ascending
        .execute()
    )

    records = res.data
    total_hours = 0.0
    clock_in_time = None


    for row in records:
        ts = format_sgt(row["timestamp"])

        if row["method"] == "CLOCK_IN":
            clock_in_time = ts
        elif row["method"] == "CLOCK_OUT":
            if clock_in_time:
                duration = ts - clock_in_time
                total_hours += duration.total_seconds() / 3600.0
                notes.append(f"Worked {round(duration.total_seconds()/3600.0, 2)} hours on {clock_in_time.date()}")
                clock_in_time = None
            else:
                notes.append(f"Unmatched CLOCK_OUT at {ts.strftime('%Y-%m-%d %H:%M:%S')} — skipped")

    # Handle ongoing session (no CLOCK_OUT yet)
    if clock_in_time:
        notes.append(f"Clocked in on {clock_in_time.strftime('%Y-%m-%d %H:%M:%S')} (no clock out)")
        # Optionally add hours up to now
        now = datetime.now(SGT)
        duration = now - clock_in_time
        total_hours += duration.total_seconds() / 3600.0

    return round(total_hours, 2)


def calculate_daily_hours(empid):
    res = (
        supabase.table("attendance")
        .select("timestamp, method")
        .eq("employee_id", empid.strip().upper())
        .order("timestamp")  # ascending
        .execute()
    )

    records = res.data
    daily_hours = defaultdict(float)
    clock_in_time = None
    notes = {}
    ongoing_sessions = {}

    for row in records:
        ts = format_sgt(row["timestamp"])
        day = ts.date()

        if row["method"] == "CLOCK_IN":
            clock_in_time = ts
        elif row["method"] == "CLOCK_OUT":
            if clock_in_time:
                duration = ts - clock_in_time
                daily_hours[day] += duration.total_seconds() / 3600.0
                notes[day] = f"{round(daily_hours[day], 2)} hours"
                clock_in_time = None
            else:
                notes[day] = "(unmatched CLOCK_OUT)"

    # Handle ongoing session (no CLOCK_OUT yet)
    if clock_in_time:
        now = datetime.now(SGT)
        duration = now - clock_in_time
        day = clock_in_time.date()
        daily_hours[day] += duration.total_seconds() / 3600.0
        notes[day] = f"{round(daily_hours[day], 2)} hours (no clock out)"
        ongoing_sessions[day] = True

    # Format breakdown
    breakdown = {}
    total_hours = 0.0
    for day, hours in daily_hours.items():
        total_hours += hours
        breakdown[str(day)] = notes.get(day, f"{round(hours, 2)} hours")

    return breakdown, round(total_hours, 2)

def log_attendance(empid_norm, method="CLOCK_IN", user_id=None):
    """Log attendance with Singapore timezone timestamp"""

    timestamp_sgt = datetime.now(SGT)
    timestamp_utc = timestamp_sgt.astimezone(timezone.utc).isoformat()

    st.write(f"user_id{user_id}, empid_norm={empid_norm}, method={method}, timestamp_utc={timestamp_utc}")
    record = {
        "user_id": user_id,
        "employee_id": empid_norm,
        "method": method,
        "timestamp": timestamp_utc
    }
    # Save to Supabase or your DB
    supabase.table("attendance").insert(record).execute()
    return record


def login_ui():
    st.subheader("Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            try:
                res = supabase_public.auth.sign_in_with_password({"email": email, "password": password})
                user = res.user
                if user:
                    st.session_state["user"] = {"id": user.id, "email": user.email}
                    st.success("Logged in")
                    time.sleep(0.8)
                    st.rerun()
            except Exception as e:
                st.error(f"Login failed: {e}")
#    with col2:
#        if st.button("Logout"):
#            supabase_public.auth.sign_out()
#            st.session_state.pop("user", None)
#            st.rerun()

# ---------------- UI: upload and capture ----------------
def upload_samples_ui():
    st.subheader("Upload face samples")
    #empid = st.text_input("Employee ID")
    #name = st.text_input("Full name")

    # --- Build employee selectbox ---
    employees_df = view_employees()
    employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()

    default_index = 0

    selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
    selected_empid = selected_label.split(" - ")[0]

    empid = selected_empid

    files = st.file_uploader("Upload multiple images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    if files and empid and st.button("Save uploaded samples"):
        empid_norm = empid.strip()
        saved = 0
        for file in files:
            try:
                pil_img = Image.open(file)
            except Exception:
                st.warning(f"Failed to open {file.name}")
                continue
            face_resized, rect, gray = preprocess_face_from_pil(pil_img)
            if face_resized is None:
                st.warning(f"No face detected in {file.name}")
                continue
            filename = f"{empid_norm}_{int(time.time())}_{os.path.splitext(file.name)[0]}.png"
            upload_face_sample_to_storage(empid_norm, filename, face_resized)
            saved += 1
        register_employee_if_missing(empid_norm, name)
        st.success(f"Uploaded {saved} samples for {selected_label.strip()} ({empid_norm})")

def capture_samples_ui():
    st.subheader("Capture face samples")

    # --- Build employee selectbox ---
    employees_df = view_employees()
    employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()

    default_index = 0

    selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
    selected_empid = selected_label.split(" - ")[0]

    empid = selected_empid
    
    # from empid display the name if exists
    #if empid:
    #    empid_norm = empid.strip()
        #st.write(f"Looking up name for {empid_norm}...")
    #    existing = supabase_public.table("users").select("name").eq("employee_id", empid_norm).execute()
    #    if existing.data:
    #        st.write
    #        name = existing.data[0].get("name", "")
    #        st.info(f"Existing name for {empid_norm}: {name}")
    #else:
    #   st.write("Enter Employee ID to lookup name.")
    #    name = ""       
    #    name = st.text_input("Full name")
    img = st.camera_input("Take a photo")

    if img and empid and st.button("Save captured sample"):
        empid_norm = empid.upper().strip()
        pil_img = Image.open(img)
        face_resized, rect, gray = preprocess_face_from_pil(pil_img)
        if face_resized is None:
            st.error("No face detected. Try again.")
            return
        filename = f"{empid_norm}_{int(time.time())}.png"
        upload_face_sample_to_storage(empid_norm, filename, face_resized)
        register_employee_if_missing(empid_norm, name)
        st.success(f"Saved sample for {name} ({empid_norm})")

def retrain_lbph_from_storage_ui():
    st.subheader("Retrain LBPH model from cloud images")
    if st.button("Retrain now"):
        try:
            ensure_tmp_dir()
            folders = list_employee_folders()
            if not folders:
                st.error("No folders found in faces bucket.")
                return

            X, y = [], []
            for folder in folders:
                empid = folder.get("name")
                if not empid:
                    continue
                files = list_files_in_folder(empid)
                for f in files:
                    fname = f.get("name")
                    if not fname:
                        continue
                    path = f"{empid}/{fname}"
                    img_bytes = download_file_bytes(FACES_BUCKET, path)
                    img_array = np.frombuffer(img_bytes, np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    face_resized = cv2.resize(img, (200, 200))
                    X.append(face_resized)
                    y.append(empid.upper().strip())
            st.write(f"Found {len(X)} samples from {len(set(y))} employees.")
            if not X:
                st.error("No valid samples found.")
                return

            try:
                res = supabase.storage.from_("models").download(LABEL_MAP_FILENAME)
                old_label_map = np.load(io.BytesIO(res), allow_pickle=True).item()
            except Exception:
                old_label_map = {}

            st.write(f"Loaded {len(old_label_map)} existing labels from Supabase.")

            # Normalize old label map into dict format {id, name}
            normalized_label_map = {}
            for emp, val in old_label_map.items():
                if isinstance(val, dict):
                    normalized_label_map[emp] = {"id": int(val["id"]), "name": val.get("name", emp)}
                else:
                    # legacy int-only format → wrap into dict
                    normalized_label_map[emp] = {"id": int(val), "name": emp}

            # Merge old and new labels
            unique = sorted(set(y))
            st.write(f"Unique labels in current data: {len(unique)}")

            label_map = normalized_label_map.copy()
            next_id = max([v["id"] for v in label_map.values()], default=-1) + 1
            st.write(f"Starting label map size: {len(label_map)}")
            st.write(f"Next label ID to assign: {next_id}")

            for emp in unique:
                if emp not in label_map:
                    label_map[emp] = {"id": next_id, "name": emp}
                    next_id += 1

            st.write(f"Merged label map size: {len(label_map)}")

            # Encode labels using the normalized dict format
            y_encoded = np.array([label_map[e]["id"] for e in y], dtype=np.int32)

            st.write(f"Total labels after merging: {len(label_map)}")


            # Load existing model if available
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            try:
                model_bytes = supabase.storage.from_("models").download(LBPH_MODEL_FILENAME)
                tmp_model_path = os.path.join(TMP_DIR, LBPH_MODEL_FILENAME)
                with open(tmp_model_path, "wb") as f:
                    f.write(model_bytes)
                recognizer.read(tmp_model_path)
                recognizer.update(X, y_encoded)   # Incremental update
            except Exception:
                recognizer.train(X, y_encoded)    # First-time training
            st.write("Model trained/updated.")
            # Save updated model locally then upload to Supabase
            tmp_model_path = os.path.join(TMP_DIR, LBPH_MODEL_FILENAME)
            recognizer.write(tmp_model_path)
            with open(tmp_model_path, "rb") as f:
                supabase.storage.from_("models").upload(LBPH_MODEL_FILENAME, f.read(), {"upsert": "true"})
            st.write("Model uploaded to Supabase.")
            # Save label map directly to Supabase
            buf = io.BytesIO()
            np.save(buf, label_map)
            buf.seek(0)
            supabase.storage.from_("models").upload(LABEL_MAP_FILENAME, buf.read(), {"upsert": "true"})
            st.write("Label map uploaded to Supabase.")
            st.success("Model and label map updated in Supabase.")
        except Exception as e:
            st.error(f"Retraining failed: {e}")

SGT = timezone(timedelta(hours=8))

def has_attendance_today(empid):
    
    empid_norm = empid

    #today = datetime.utcnow().date()
    today = datetime.now(SGT).date()
    st.write(f"Checking attendance for {empid_norm} on {today.isoformat()}")
    #start = datetime.combine(today, datetime.min.time()).isoformat() + "Z"
    #end = datetime.combine(today, datetime.max.time()).isoformat() + "Z"
    start = datetime.combine(today, datetime.min.time(), tzinfo=SGT).isoformat()
    end = datetime.combine(today, datetime.max.time(), tzinfo=SGT).isoformat()

    res = (
        supabase_public.table("attendance")
        .select("id")
        .eq("employee_id", empid_norm)
        .gte("timestamp", start)   # or use your actual timestamp column name
        .lte("timestamp", end)
        .execute()
    )
    st.write(f"Attendance records found: {len(res.data)}")    
    return len(res.data) > 0

#def get_user(empid):
#    empid_norm = empid
#    res = supabase_public.table("users").select("*").eq("employee_id", empid_norm).execute()
#    if res.data:
#        return res.data[0]
#    return None

def get_user(empid):
    empid_norm = str(empid).strip()  # normalize to string
    res = supabase_public.table("users").select("*").eq("employee_id", empid_norm).execute()
    #st.write("DEBUG Supabase response:", res.data)

    if res.data and len(res.data) > 0:
        return res.data[0]   # returns a dict
    return None



# ---------------- Upload image to cloud for retraining ----------------
def upload_image_to_cloud(img, empid, method):
    """Upload captured image to Supabase Storage for retraining."""
    empid_norm = empid.upper().strip()
    pil_img = Image.open(img)
    face_resized, rect, gray = preprocess_face_from_pil(pil_img)
    if face_resized is None:
        st.error("No face detected in the image. Cannot upload for retraining.")
        return
    filename = f"{empid_norm}_{int(time.time())}_{method}.png"
    upload_face_sample_to_storage(empid_norm, filename, face_resized)
    st.info(f"Uploaded image for {empid_norm} to cloud storage for retraining.")


# ---------------- Get Attendance for today ----------------
def get_attendance_for_today(empid):
    """Fetch all attendance records for the given employee on today's date."""
    today = datetime.now(SGT).date()

    res = (
        supabase.table("attendance")
        .select("timestamp, method")
        .eq("employee_id", empid)
        .order("timestamp")
        .execute()
    )

    records = res.data

    # Filter only today's records
    todays_records = [
        r for r in records if format_sgt(r["timestamp"]).date() == today
    ]

    return todays_records
# ---------------- View Employees ----------------
def view_employees():

    res = supabase_public.table("users").select("*").execute()
    data = res.data
    df = pd.DataFrame(data)
    return df


# ---------------- Attendance ----------------
def attendance_check_ui():
    st.subheader("Mark attendance (camera)")
    img = st.camera_input("Take a photo")
    if not img:
        return

    recognizer, label_map = load_model_from_storage()
    if recognizer is None:
        return

    pil_img = Image.open(img)
    gray = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(60, 60))
    if len(faces) == 0:
        st.warning("No face detected.")
        return

    (x, y, w, h) = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)[0]
    roi = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(roi, (200, 200))
    label_id, confidence = recognizer.predict(face_resized)

    inv_map = {v["id"]: k for k, v in label_map.items()}
    recognized_empid = inv_map.get(label_id)

    # --- Build employee selectbox ---
    employees_df = view_employees()
    #employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()
    employee_options = employees_df.apply(
        lambda row: f"{row['employee_id']} - {row['name']}", axis=1).values.tolist()

    default_index = 0
    if recognized_empid:
        user = get_user(recognized_empid) or {}
        recognized_label = f"{recognized_empid} - {user.get('name', 'Unknown')}"
        if recognized_label in employee_options:
            default_index = employee_options.index(recognized_label)
            st.info(f"Recognized {recognized_label} (confidence {confidence:.2f}). You may override below if incorrect.")
    else:
        st.warning("Face not recognized. Please select employee manually.")

    #selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
    #selected_empid = int(selected_label.split(" - ")[0])
    #selected_empid = selected_label.split(" - ")[0]
    #selected_user = get_user(selected_empid)

    selected_label = st.selectbox("Select Employee", employee_options, index=default_index) if employee_options else None

    if selected_label:
        selected_empid = selected_label.split(" - ")[0]
        empid = selected_empid
    else:
        empid = None
        st.warning("No employees available to select.")


    # --- Attendance logic unified ---
    attendance_today = get_attendance_for_today(selected_empid)
    has_clock_in = any(r["method"] == "CLOCK_IN" for r in attendance_today)
    has_clock_out = any(r["method"] == "CLOCK_OUT" for r in attendance_today)

    if not has_clock_in:
        st.info(f"{selected_empid} has not clocked in yet today. Next action will be CLOCK_IN.")
        if st.button(f"Confirm CLOCK_IN for {selected_label}"):
            #st.write("DEBUG: selected_empid =", selected_empid)
            selected_user = get_user(selected_empid)
            log_attendance(selected_empid, method="CLOCK_IN", user_id=selected_user["id"])
            st.success(f"Clock-in marked for {selected_label}")
            upload_image_to_cloud(img, selected_empid, "CLOCK_IN")

    elif has_clock_in and not has_clock_out:
        st.info(f"{selected_empid} already clocked in today. Next action will be CLOCK_OUT.")
        if st.button(f"Confirm CLOCK_OUT for {selected_label}"):
            selected_user = get_user(selected_empid)
            log_attendance(selected_empid, method="CLOCK_OUT", user_id=selected_user["id"])
            st.success(f"Clock-out recorded for {selected_label}")
            upload_image_to_cloud(img, selected_empid, "CLOCK_OUT")

    elif has_clock_in and has_clock_out:
        st.info(f"{selected_empid} has already clocked in and clocked out today. No further action needed.")

def attendance_manual_ui():
    st.subheader("Manual attendance (fallback)")

    # Fetch all employees from users table
    employees = supabase_public.table("users").select("employee_id, name, id").execute().data

    if not employees:
        st.warning("No employees found. Please register employees first.")
        return

    # Build dropdown options like "001 - Jemma"
    options = {f"{emp['employee_id']} - {emp['name']}": emp for emp in employees}

    # Insert default option at the top
    labels = ["Select employee"] + list(options.keys())

    selected_label = st.selectbox("Select Employee", labels)

    if selected_label == "Select employee":
        st.info("Please choose an employee from the dropdown.")
        return

    # Get the selected employee record
    selected_emp = options[selected_label]
    empid_norm = selected_emp["employee_id"]
    name = selected_emp["name"]

    st.info(f"Selected: {name} ({empid_norm})")

    # Get today's records
    attendance_today = get_attendance_for_today(empid_norm)
    has_clock_in = any(r["method"] == "CLOCK_IN" for r in attendance_today)
    has_clock_out = any(r["method"] == "CLOCK_OUT" for r in attendance_today)

    if not has_clock_in:
        st.info(f"{name} ({empid_norm}) has not clocked in yet today. Next action will be CLOCK_IN.")
        if st.button(f"Confirm CLOCK_IN for {name} ({empid_norm})"):
            log_attendance(empid_norm, method="CLOCK_IN", user_id=selected_emp["id"])
            st.success(f"Clock-in marked for {empid_norm} (manual)")

    elif has_clock_in and not has_clock_out:
        st.info(f"{name} ({empid_norm}) already clocked in today. Next action will be CLOCK_OUT.")
        if st.button(f"Confirm CLOCK_OUT for {name} ({empid_norm})"):
            log_attendance(empid_norm, method="CLOCK_OUT", user_id=selected_emp["id"])
            st.success(f"Clock-out recorded for {empid_norm} (manual)")

    elif has_clock_in and has_clock_out:
        st.info(f"{name} ({empid_norm}) has already clocked in and clocked out today. No further action needed.")


def attendance_manual_ui2():
    st.subheader("Manual attendance (fallback)")
    empid = st.text_input("Employee ID")
    # show name of the employee if exists
    if empid:   
        empid_norm = empid.strip()
        st.write(f"Looking up name for {empid_norm}...")
        existing = supabase_public.table("users").select("name").eq("employee_id", empid_norm).execute()
        if existing.data:
            user = get_user(empid)
            #user = existing.data[0]
            name = user.get("name", "Unknown")

            st.info(f"Existing name for {empid_norm}: {name}")
            
            # Get today's records
            attendance_today = get_attendance_for_today(empid_norm)
            has_clock_in = any(r["method"] == "CLOCK_IN" for r in attendance_today)
            has_clock_out = any(r["method"] == "CLOCK_OUT" for r in attendance_today)

            if not has_clock_in:
                st.info(f"{name} ({empid_norm}) has not clocked in yet today. Next action will be CLOCK_IN.")
                if st.button(f"Confirm CLOCK_IN for {name} ({empid_norm})"):
                    log_attendance(empid_norm, method="CLOCK_IN", user_id=user["id"])
                    st.success(f"Clock-in marked for {empid_norm} (manual)")

            elif has_clock_in and not has_clock_out:
                st.info(f"{name} ({empid_norm}) already clocked in today. Next action will be CLOCK_OUT.")
                if st.button(f"Confirm CLOCK_OUT for {name} ({empid_norm})"):
                    log_attendance(empid_norm, method="CLOCK_OUT", user_id=user["id"])
                    st.success(f"Clock-out recorded for {empid_norm} (manual)")

            elif has_clock_in and has_clock_out:
                st.info(f"{name} ({empid_norm}) has already clocked in and clocked out today. No further action needed.")

        else:
            st.warning("Employee does not exist. Please register first.")

def verify_lbph_labels_with_counts(min_images=10):
    """List all employees in the LBPH model, their image counts, and warn if below threshold.
       Reads label_map.npy directly from Supabase storage instead of local."""
    try:
        # Try to download label map from Supabase
        try:
            res = supabase.storage.from_("models").download(LABEL_MAP_FILENAME)
            label_map = np.load(io.BytesIO(res), allow_pickle=True).item()
        except Exception:
            st.warning("No label map found in Supabase. Train or retrain the model first.")
            return

        st.subheader("Employees in Current Model")
        counts = {}

        # Count images per employee by checking Supabase folders
        for emp in label_map.keys():
            files = list_files_in_folder(emp)  # helper that lists files in Supabase bucket
            counts[emp] = len(files)

        # Show total employees
        st.info(f"Total employees in model: {len(label_map)}")

        for emp, data in label_map.items():
            idx = data["id"] if isinstance(data, dict) else data
            name = data["name"] if isinstance(data, dict) else emp

            count = counts.get(emp, 0)
            if count < min_images:
                st.warning(f"ID {idx}: {emp} ({name}) → {count} images (⚠️ below recommended {min_images})")
            else:
                st.write(f"ID {idx}: {emp} ({name}) → {count} images")

    except Exception as e:
        st.error(f"Verification failed: {e}")

def register_employee(empid, name):
    empid = empid.strip()
    name = name.upper().strip()
    supabase_admin.table("users").insert({"employee_id": empid, "name": name}).execute()

# ---------------- UI: Delete Employee ----------------
def delete_employee_ui():
    st.subheader("Delete employee")
    # --- Build employee selectbox ---
    employees_df = view_employees()
    employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()

    default_index = 0

    selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
    selected_empid = selected_label.split(" - ")[0]
    empid = selected_empid

    if st.button("Delete"):
        if empid:
            empid_norm = empid.strip()
            supabase_admin.table("users").delete().eq("employee_id", empid_norm).execute()
            st.success(f"Deleted employee {empid_norm}")
        else:
            st.error("Please select an Employee ID.")


def register_employee_ui():
    st.subheader("Register employee")
    empid = st.text_input("Employee ID")
    name = st.text_input("Full name")
    if st.button("Register"):
        if empid and name:
            register_employee(empid, name)
            st.success(f"Registered {empid.strip()} - {name}")
        else:
            st.error("Please enter both Employee ID and Name.")

# ---------------- Main ----------------
def main():
    st.title("Attendance with Face Recognition (Supabase + Streamlit)")

    # Use your timezone helper if needed
    today = datetime.now(SGT).date()   # ✅ define 'today'

    # First day of current month
    first_day = today.replace(day=1)

    # Last day of current month
    # Trick: go to first day of next month, subtract 1 day
    if today.month == 12:
        next_month = today.replace(year=today.year+1, month=1, day=1)
    else:
        next_month = today.replace(month=today.month+1, day=1)
    last_day = next_month - timedelta(days=1)

    # Auth gate (anon client)
    user = st.session_state.get("user")
    if not user:
        login_ui()
        st.info("Please log in to continue.")
        return

    #st.sidebar.markdown(f"Signed in as: {user.get('email','')} UUID: {st.session_state["user"]["id"]}")
    st.sidebar.markdown(f"Signed in as: {user.get('email','')}")

    #res = supabase.table("users").select("*").execute()
    #print(res.data)
    if "user" in st.session_state:
        if st.sidebar.button("Logout"):
            supabase_public.auth.sign_out()
            st.session_state.pop("user", None)
            st.rerun()

    page = st.sidebar.radio(
        "Menu",
        [
            "Register Employee",
            "Delete Employee",
            "Upload samples",
            "Capture samples",
            "Retrain model",
            "Verify Employees",
            "Mark attendance",
            "Manual attendance",
            "Daily Timesheet",
            "Monthly Timesheet",
            "View Attendance"
        ],
    )
    if page == "Register Employee":
        register_employee_ui()
    if page == "Delete Employee":
        delete_employee_ui()
    if page == "Upload samples":
        upload_samples_ui()
    elif page == "Capture samples":
        capture_samples_ui()
    elif page == "Retrain model":
        retrain_lbph_from_storage_ui()
    elif page == "Mark attendance":
        attendance_check_ui()
    elif page == "Manual attendance":
        attendance_manual_ui()
    elif page == "Verify Employees":
        verify_lbph_labels_with_counts()
    elif page == "Daily Timesheet":
        st.subheader("Daily Timesheet")

        # --- Build employee selectbox ---
        employees_df = view_employees()
        employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()

        default_index = 0

        selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
        selected_empid = selected_label.split(" - ")[0]
        empid = selected_empid

        if st.button("Get Daily Timesheet"):
            if empid.strip():
                daily_breakdown, total_hours = calculate_daily_hours(empid)

                st.subheader(f"Daily hours worked for {selected_label.strip()}")
                for day, hours in daily_breakdown.items():
                    st.write(f"{day}: {hours}")

                st.markdown(f"**Grand Total: {total_hours} hours**")
            else:
                st.warning("Enter an Employee ID.")
    elif page == "Monthly Timesheet":
        st.subheader("Monthly Timesheet")

        # --- Build employee selectbox ---
        employees_df = view_employees()
        employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()

        default_index = 0

        selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
        selected_empid = selected_label.split(" - ")[0]
  
        empid = selected_empid
        if st.button("Get Monthly Timesheet"):
            if empid.strip():
                notes = []
                total_hours = calculate_total_hours(empid, notes)
                st.subheader(f"Total hours worked for {selected_label.strip()}")
                st.write(f"Total hours worked for {selected_label.strip()}: {total_hours} hours")
                for note in notes:
                    st.write(note)
            else:
                st.warning("Enter an Employee ID.")
    elif page == "View Attendance":
        st.subheader("View Attendance Records")

        # --- Build employee selectbox ---
        employees_df = view_employees()
        employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()
        default_index = 0

        selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
        selected_empid = selected_label.split(" - ")[0]
        empid = selected_empid

        # Default date range: current month
        today = datetime.now(SGT).date()
        first_day = today.replace(day=1)
        if today.month == 12:
            next_month = today.replace(year=today.year+1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month+1, day=1)
        last_day = next_month - timedelta(days=1)

        col1, col2 = st.columns(2)
        start_date = col1.date_input("From date", value=first_day)
        end_date = col2.date_input("To date", value=last_day)

        if st.button("View Attendance"):
            if empid.strip():
                res = (
                    supabase_public.table("attendance")
                    .select("timestamp, method")
                    .eq("employee_id", empid.strip().upper())
                    .order("timestamp")
                    .execute()
                )
                records = res.data

                if records:
                    df = pd.DataFrame(records)
                    df["timestamp"] = df["timestamp"].apply(format_sgt)
                    df = df.sort_values("timestamp")

                    summary = []
                    clock_in = None

                    for _, row in df.iterrows():
                        if row["method"] == "CLOCK_IN":
                            if clock_in is not None:
                                # Previous clock-in never got a clock-out → mark incomplete
                                summary.append({
                                    "Date": clock_in.date(),
                                    "CLOCKIN": clock_in.strftime("%H:%M:%S"),
                                    "CLOCKOUT": "",
                                    "Hours": None
                                })
                            clock_in = row["timestamp"]

                        elif row["method"] == "CLOCK_OUT" and clock_in is not None:
                            clock_out = row["timestamp"]
                            duration = clock_out - clock_in
                            hours = round(duration.total_seconds() / 3600.0, 2)
                            summary.append({
                                "Date": clock_in.date(),
                                "CLOCKIN": clock_in.strftime("%H:%M:%S"),
                                "CLOCKOUT": clock_out.strftime("%H:%M:%S"),
                                "Hours": hours
                            })
                            clock_in = None

                    # Handle last unmatched clock-in
                    if clock_in is not None:
                        summary.append({
                            "Date": clock_in.date(),
                            "CLOCKIN": clock_in.strftime("%H:%M:%S"),
                            "CLOCKOUT": "",
                            "Hours": None
                        })

                    summary_df = pd.DataFrame(summary)

                    # Apply date filter
                    mask = (summary_df["Date"] >= start_date) & (summary_df["Date"] <= end_date)
                    summary_df = summary_df.loc[mask]

                    st.dataframe(summary_df)

                    # Clean Hours column
                    summary_df["Hours"] = pd.to_numeric(summary_df["Hours"], errors="coerce").fillna(0)

                    # Compute total
                    total_hours = summary_df["Hours"].sum()
                    st.write(f"**Grand Total Hours: {round(total_hours, 2)}**")

                else:
                    st.info(f"No attendance records found for {empid.strip()}.")
            else:
                st.warning("Enter an Employee ID.")

    elif page == "View Attendance2":
        st.subheader("View Attendance Records")
        # dropdown to select empid from users table with empid - name format

        # --- Build employee selectbox ---
        employees_df = view_employees()
        employee_options = employees_df.apply(lambda row: f"{row['employee_id']} - {row['name']}", axis=1).tolist()

        default_index = 0

        selected_label = st.selectbox("Select Employee", employee_options, index=default_index)
        selected_empid = selected_label.split(" - ")[0]
        empid = selected_empid

        # Default date range: current month
        today = datetime.now(SGT).date()
        first_day = today.replace(day=1)
        if today.month == 12:
            next_month = today.replace(year=today.year+1, month=1, day=1)
        else:
            next_month = today.replace(month=today.month+1, day=1)
        last_day = next_month - timedelta(days=1)

        col1, col2 = st.columns(2)
        start_date = col1.date_input("From date", value=first_day)
        end_date = col2.date_input("To date", value=last_day)

        if st.button("View Attendance"):
            if empid.strip():
                res = (
                    supabase_public.table("attendance")
                    .select("timestamp, method")
                    .eq("employee_id", empid.strip().upper())
                    .order("timestamp")
                    .execute()
                )
                records = res.data

                if records:
                    df = pd.DataFrame(records)
                    df["timestamp"] = df["timestamp"].apply(format_sgt)
                    df["date"] = df["timestamp"].dt.date

                    # Group by date: first CLOCK_IN and last CLOCK_OUT
                    summary = []
                    for d, group in df.groupby("date"):
                        clock_in = group.loc[group["method"] == "CLOCK_IN", "timestamp"].min()
                        clock_out = group.loc[group["method"] == "CLOCK_OUT", "timestamp"].max()

                        # Calculate hours if both exist
                        #hours = ""
                        hours = np.nan

                        if pd.notnull(clock_in) and pd.notnull(clock_out):
                            duration = clock_out - clock_in
                            hours = round(duration.total_seconds() / 3600.0, 2)

                        summary.append({
                            "Date": d,
                            "CLOCKIN": clock_in.strftime("%H:%M:%S") if pd.notnull(clock_in) else "",
                            "CLOCKOUT": clock_out.strftime("%H:%M:%S") if pd.notnull(clock_out) else "",
                            "Hours": hours
                        })

                    summary_df = pd.DataFrame(summary)

                    # Apply date filter
                    mask = (summary_df["Date"] >= start_date) & (summary_df["Date"] <= end_date)
                    summary_df = summary_df.loc[mask]

                    st.dataframe(summary_df)
                else:
                    st.info(f"No attendance records found for {empid.strip()}.")
            else:
                st.warning("Enter an Employee ID.")

            # Clean Hours column once
            summary_df["Hours"] = pd.to_numeric(summary_df["Hours"].mask(summary_df["Hours"] == "", 0), errors="coerce")

            # Compute total
            total_hours = summary_df["Hours"].sum()

            st.write(f"**Grand Total Hours: {round(total_hours, 2)}**")



if __name__ == "__main__":
    main()