import os
import subprocess

input_folder = "./DME_Study_OCT"
output_root = "./frames"
video_extensions = ('.avi', '.mp4', '.mov', '.mkv')

os.makedirs(output_root, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.lower().endswith(video_extensions):
        input_path = os.path.join(input_folder, filename)
        video_name = os.path.splitext(filename)[0]
        fixed_video_path = os.path.join(input_folder, f"{video_name}_fixed.mp4")
        output_dir = os.path.join(output_root, video_name)
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Re-encode video (especially for .avi files)
        print(f"Re-encoding {filename}...")
        subprocess.run([
            "ffmpeg", "-fflags", "+genpts",
            "-i", input_path,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-an",
            fixed_video_path
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        # Step 2: Extract frames from re-encoded video
        print(f"Extracting frames from: {video_name}_fixed.mp4")
        output_pattern = os.path.join(output_dir, "frame_%03d.jpg")
        subprocess.run([
            "ffmpeg",
            "-i", fixed_video_path,
            "-vf", "fps=5",  # adjust fps here if needed
            output_pattern
        ], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
