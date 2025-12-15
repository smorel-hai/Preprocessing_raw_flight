from scipy.spatial.transform import Rotation as R_scipy


from pathlib import Path
import shutil


def get_euler_angles_scipy(R_matrix):
    """
    Returns Yaw, Pitch, Roll in degrees.
    Convention: 'xyz' (extrinsic).
    """
    # Create a rotation object from the matrix
    r = R_scipy.from_matrix(R_matrix)

    # Convert to Euler angles
    # 'xyz' is the standard convention for cameras
    yaw, pitch, roll = r.as_euler('xyz', degrees=True)

    return yaw, pitch, roll


def transfer_skip_existing_names(source_folder, target_folder, skip_parents=['global_tiles']):
    """
    Recursively moves files from source to target using pathlib.
    Skips files if the name already exists in the destination.
    """
    # Convert strings to Path objects
    source = Path(source_folder)
    target = Path(target_folder)

    # Create target directory if it doesn't exist
    target.mkdir(parents=True, exist_ok=True)

    files_copied = 0
    files_skipped = 0

    print(f"--- Transferring: {source} -> {target} ---")

    # rglob('*') recursively iterates over all files and folders
    for src_file in source.rglob('*'):
        if src_file.is_file():
            #  Prevent the copy of certain files
            if src_file.parent.name in skip_parents:
                continue
            # Calculate the relative path (e.g. 'subfolder/image.jpg')
            relative_path = src_file.relative_to(source)

            # Construct the full destination path
            # (pathlib allows using '/' operator to join paths)
            dest_file = target / relative_path

            # --- THE CHECK ---
            if dest_file.exists():
                print(f"[SKIP] {dest_file.name}")
                files_skipped += 1
            else:
                # Ensure the specific sub-folder exists before copying
                dest_file.parent.mkdir(parents=True, exist_ok=True)

                try:
                    src_file.rename(dest_file)
                    print(f"[MOVE] {dest_file.name}")
                    files_copied += 1
                except Exception as e:
                    print(f"[ERROR] {src_file.name}: {e}")

    print(f"\n--- Done ---")
    print(f"Copied:  {files_copied}")
    print(f"Skipped: {files_skipped}")


def delete_folder(folder_path):
    """
    Deletes a folder and all its contents.
    """
    target = Path(folder_path)

    # 1. Check if the folder actually exists
    if not target.exists():
        print(f"[Error] The folder '{folder_path}' does not exist.")
        return

    # 2. Check if it is actually a directory (not a file)
    if not target.is_dir():
        print(f"[Error] '{folder_path}' is a file, not a folder.")
        return

    # 3. Attempt to delete
    try:
        # shutil.rmtree is the function that handles recursive deletion
        shutil.rmtree(target)
        print(f"[Success] Deleted folder: {folder_path}")
    except OSError as e:
        print(f"[Error] Failed to delete {folder_path}. Reason: {e}")


def find_values_gen(data, target_key):
    if isinstance(data, dict):
        for key, value in data.items():
            if key == target_key:
                yield value
            # Recursively yield from children
            yield from find_values_gen(value, target_key)

    elif isinstance(data, list):
        for item in data:
            yield from find_values_gen(item, target_key)
