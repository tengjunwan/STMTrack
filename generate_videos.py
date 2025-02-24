from moviepy import ImageSequenceClip
from pathlib import Path
from tqdm import tqdm


input_folder = Path("/media/tengjunwan/新加卷/ObjectTracing/stark/Stark-main/data/lasot/votcar/votcar-1/img")
output_video = Path("./video/votcar-1.mp4")

frame_rate = 30

image_paths = [str(i) for i in input_folder.glob("*.jpg")]
print(f"images num: {len(image_paths)}")

clip = ImageSequenceClip(image_paths, fps=frame_rate)
clip.write_videofile(output_video, codec="libx264")

print("done")