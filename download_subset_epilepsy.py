import os
import subprocess

BASE_S3 = "s3://openneuro.org/ds005602"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ds005602")

# 20 epilepsy patients (sub-1 to sub-442 range) + 20 healthy controls (sub-4001 to sub-4100 range)
subjects = [
    "sub-1", "sub-2", "sub-3", "sub-4", "sub-5",
    "sub-6", "sub-7", "sub-8", "sub-10", "sub-11",
    "sub-12", "sub-13", "sub-14", "sub-16", "sub-17",
    "sub-18", "sub-19", "sub-21", "sub-22", "sub-23",
    "sub-4001", "sub-4002", "sub-4003", "sub-4004", "sub-4005",
    "sub-4006", "sub-4007", "sub-4008", "sub-4009", "sub-4011",
    "sub-4012", "sub-4013", "sub-4015", "sub-4016", "sub-4018",
    "sub-4019", "sub-4020", "sub-4021", "sub-4023", "sub-4024",
]

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"Saving to: {LOCAL_DIR}")
print(f"Total subjects: {len(subjects)}\n")

for sub in subjects:
    print(f"Downloading: {sub}")
    local_anat = os.path.join(LOCAL_DIR, sub, "anat")
    os.makedirs(local_anat, exist_ok=True)

    result = subprocess.run([
        "python3", "-m", "awscli", "s3", "sync",
        "--no-sign-request",
        f"{BASE_S3}/{sub}/anat",
        local_anat,
        "--exclude", "*",
        "--include", "*T1w.nii.gz",
        "--include", "*FLAIR.nii.gz"
    ])

    if result.returncode != 0:
        print(f"  WARNING: download failed for {sub}")
    else:
        files = os.listdir(local_anat)
        print(f"  Files: {files}")

print("\nDownload complete.")
print(f"Dataset at: {LOCAL_DIR}")
