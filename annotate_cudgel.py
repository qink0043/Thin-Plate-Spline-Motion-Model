import cv2
import numpy as np
import imageio
import json
import os
from argparse import ArgumentParser
from tqdm import tqdm

class Annotator:
    def __init__(self, video_path, output_path, img_shape=(256, 256)):
        self.video_path = video_path
        self.output_path = output_path
        self.img_shape = img_shape
        self.frames = []
        self.annotations = []
        self.current_frame_idx = 0
        self.temp_points = []
        self.window_name = "Cudgel Annotator - [N]ext, [P]rev, [S]ave, [Q]uit, [C]opy Prev"
        
        self.load_video()
        
        # Try to load existing annotations
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r') as f:
                    self.annotations = json.load(f)
                print(f"Loaded {len(self.annotations)} existing annotations.")
                self.current_frame_idx = min(len(self.annotations), len(self.frames) - 1)
            except Exception as e:
                print(f"Error loading existing annotations: {e}")
                self.annotations = []

    def load_video(self):
        print(f"Loading video: {self.video_path}")
        reader = imageio.get_reader(self.video_path)
        for im in tqdm(reader, desc="Reading Frames"):
            # Resize to match model expectation
            frame = cv2.resize(im, (self.img_shape[1], self.img_shape[0]))
            self.frames.append(frame)
        reader.close()
        print(f"Loaded {len(self.frames)} frames.")

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.temp_points) < 2:
                self.temp_points.append([x, y])
                self.update_display()

    def update_display(self):
        frame = self.frames[self.current_frame_idx].copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Draw existing points for this frame
        points_to_draw = []
        if len(self.temp_points) > 0:
            points_to_draw = self.temp_points
        elif self.current_frame_idx < len(self.annotations):
            points_to_draw = self.annotations[self.current_frame_idx]

        for i, pt in enumerate(points_to_draw):
            color = (0, 0, 255) if i == 0 else (0, 255, 0) # Red then Green
            cv2.circle(frame, tuple(map(int, pt)), 4, color, -1)
        
        if len(points_to_draw) == 2:
            cv2.line(frame, tuple(map(int, points_to_draw[0])), tuple(map(int, points_to_draw[1])), (255, 255, 0), 2)

        cv2.putText(frame, f"Frame: {self.current_frame_idx}/{len(self.frames)-1}", (10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if self.current_frame_idx < len(self.annotations):
            cv2.putText(frame, "STATUS: ANNOTATED", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "STATUS: NEED INPUT", (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(self.window_name, frame)

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        while True:
            self.update_display()
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): # Quit
                break
            elif key == ord('n'): # Next frame
                if len(self.temp_points) == 2:
                    if self.current_frame_idx < len(self.annotations):
                        self.annotations[self.current_frame_idx] = self.temp_points
                    else:
                        self.annotations.append(self.temp_points)
                    self.temp_points = []
                    self.current_frame_idx = min(self.current_frame_idx + 1, len(self.frames) - 1)
                elif self.current_frame_idx < len(self.annotations):
                     self.current_frame_idx = min(self.current_frame_idx + 1, len(self.frames) - 1)
                else:
                    print("Please select 2 points first!")
            elif key == ord('p'): # Previous frame
                self.current_frame_idx = max(self.current_frame_idx - 1, 0)
                self.temp_points = []
            elif key == ord('c'): # Copy previous frame points
                if self.current_frame_idx > 0 and self.current_frame_idx - 1 < len(self.annotations):
                    self.temp_points = self.annotations[self.current_frame_idx - 1].copy()
            elif key == ord('s'): # Save
                with open(self.output_path, 'w') as f:
                    json.dump(self.annotations, f)
                print(f"Saved {len(self.annotations)} frames to {self.output_path}")
            elif key == 27: # ESC to clear current frame
                self.temp_points = []

        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = ArgumentParser(description="Cudgel Annotation Tool")
    parser.add_argument("--video", required=True, help="Path to driving video")
    parser.add_argument("--output", default="annotations.json", help="Output JSON path")
    parser.add_argument("--img_shape", default="256,256", help="Image shape H,W")
    
    args = parser.parse_args()
    h, w = map(int, args.img_shape.split(','))
    
    annotator = Annotator(args.video, args.output, (h, w))
    annotator.run()
