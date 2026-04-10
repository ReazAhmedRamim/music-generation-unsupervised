# setup_datasets.py
import os, shutil, glob

# ══════════════════════════════════════════════════
# CHANGE THESE TO YOUR ACTUAL PATHS
# ══════════════════════════════════════════════════
MAESTRO_DIR = r"D:\onedrive\Desktop\cse425\ds - 1 - maestro\maestro-v3.0.0"
GROOVE_DIR  = r"D:\onedrive\Desktop\cse425\ds - 3 - groove\groove"
LAKH_DIR    = r"D:\onedrive\Desktop\cse425\ds - 2\archive"
# ══════════════════════════════════════════════════

OUTPUT_BASE = "data/raw_midi"

def copy_files(src_glob, dest_folder, max_files=None, label=""):
    os.makedirs(dest_folder, exist_ok=True)
    files = glob.glob(src_glob, recursive=True)
    if max_files:
        files = files[:max_files]
    copied = 0
    for f in files:
        try:
            shutil.copy(f, dest_folder)
            copied += 1
        except:
            pass
    print(f"  [{label}] Copied {copied} files → {dest_folder}")
    return copied

print("=" * 50)
print("  Dataset Setup")
print("=" * 50)

# MAESTRO → classical
print("\n[1/3] MAESTRO → classical ...")
copy_files(os.path.join(MAESTRO_DIR, "**", "*.midi"),
           os.path.join(OUTPUT_BASE, "classical"), label="MAESTRO .midi")
copy_files(os.path.join(MAESTRO_DIR, "**", "*.mid"),
           os.path.join(OUTPUT_BASE, "classical"), label="MAESTRO .mid")

# GROOVE → jazz
print("\n[2/3] GROOVE → jazz ...")
copy_files(os.path.join(GROOVE_DIR, "**", "*.mid"),
           os.path.join(OUTPUT_BASE, "jazz"), label="Groove")

# LAKH → rock / pop / electronic
print("\n[3/3] LAKH → rock / pop / electronic ...")
lakh_files = glob.glob(os.path.join(LAKH_DIR, "**", "*.mid"), recursive=True)
print(f"  Found {len(lakh_files)} Lakh files")

if len(lakh_files) == 0:
    print("  ERROR: No files found! Check LAKH_DIR path")
else:
    chunk = len(lakh_files) // 3
    splits = {
        "rock":       lakh_files[0       : chunk],
        "pop":        lakh_files[chunk   : chunk*2],
        "electronic": lakh_files[chunk*2 :],
    }
    for genre, files in splits.items():
        dest = os.path.join(OUTPUT_BASE, genre)
        os.makedirs(dest, exist_ok=True)
        copied = 0
        for f in files:
            try:
                shutil.copy(f, dest)
                copied += 1
            except:
                pass
        print(f"  [{genre}] Copied {copied} files → {dest}")

# Final summary
print("\n" + "=" * 50)
print("  Final Count:")
for genre in ["classical", "jazz", "rock", "pop", "electronic"]:
    folder = os.path.join(OUTPUT_BASE, genre)
    count  = len(glob.glob(os.path.join(folder, "*.mid")))  + \
             len(glob.glob(os.path.join(folder, "*.midi")))
    status = "✓" if count > 0 else "✗ EMPTY — check path"
    print(f"  {status}  {genre:<12}  {count} files")
print("=" * 50)
print("\nDone! Now run: python src/preprocessing/midi_parser.py")