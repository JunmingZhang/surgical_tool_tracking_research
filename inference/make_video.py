from distutils import extension
import cv2
import os

# adapted from https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
def make_video(image_folder, video_name, num_frames, extension="png"):
    images = [img for img in os.listdir(image_folder) if img.endswith("." + extension)][:num_frames]
    images = sorted(images, key=lambda x: int(x.split('.')[0][4:]))
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, 0, 10, (width,height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        print(image)

    cv2.destroyAllWindows()
    video.release()

if __name__ == "__main__":
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\train\RGB5d5f3c3b-cf26-44a1-b7ba-7c4f065608b0"), "videos/train_left.avi", 100, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\train\RGB063986f6-897a-473e-8cea-3508d887306d"), "videos/train_right.avi", "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\train\RGB5d5f3c3b-cf26-44a1-b7ba-7c4f065608b0"), "videos/train_left.avi", 1000, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\unity_lily_1000\RGB4531aaa9-aa26-40aa-8555-d8b69bd36f4a"), "videos/lily_1000.avi", 1000, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\sim_real_test\RGBe4f9428c-9408-48b5-81b7-a545658c1560"), "videos/sim_real_1000.avi", 1000, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\sim_real_test2\RGB48eec5a1-2415-4271-a06f-7be317f00aaa"), "videos/sim_real_test2.avi", 1000, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\sim_real_test3\RGBf49609c1-9103-4d74-a88a-2b814e4000ea"), "videos/sim_real_test3.avi", 1000, "png")
    # make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\sim_real_test4\RGB586a29aa-f6be-4e8a-994f-3492d51bac60"), "videos/sim_real_test4.avi", 1000, "png")
    make_video(os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\sim_real_test5\RGB1ae570d1-27d0-43fe-af43-edeca650d476"), "videos/sim_real_test5.avi", 1000, "png")
    