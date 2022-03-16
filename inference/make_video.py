from distutils import extension
import cv2
import os

# adapted from https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
def make_video(image_folder, video_name, num_frames, extension="png"):
    images = [img for img in os.listdir(image_folder) if img.endswith("." + extension)][:num_frames]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\train\RGB5d5f3c3b-cf26-44a1-b7ba-7c4f065608b0"), "videos/train_left.avi", 100, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\train\RGB063986f6-897a-473e-8cea-3508d887306d"), "videos/train_right.avi", "png")
    