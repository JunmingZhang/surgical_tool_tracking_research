import deeplabcut
import os

# create the project
deeplabcut.create_new_project("surgical_tool_tracking", "junming_zhang", \
    [os.path.abspath("C:\\Users\\peter\\AppData\\LocalLow\\DefaultCompany\\surgical_tool_tracking_research\\inference\\videos"), \
        os.path.abspath("C:\\Users\\peter\\AppData\\LocalLow\\DefaultCompany\\surgical_tool_tracking_research\\inference\\videos")], \
            copy_videos=True)