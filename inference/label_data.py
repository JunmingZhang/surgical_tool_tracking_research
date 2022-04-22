# Reference to https://mcsscm.utm.utoronto.ca/medcvr/medcvr-ml/-/blob/feature/deeplabcut_keypoint_localization/medcvr_ml/deeplabcut/get_labelled_dataframe.py
# A script for loading labels to train the network
# get collectedData.h5 and .csv from unity simulated img frames and keypoint labels

# System Imports
import os
import numpy as np
import pandas as pd

from datasetinsights.datasets.unity_perception import AnnotationDefinitions
# from datasetinsights.datasets.unity_perception import MetricDefinitions
from datasetinsights.datasets.unity_perception.captures import Captures

# adapted from Mustafa's work: from https://mcsscm.utm.utoronto.ca/medcvr/medcvr-ml/-/blob/dev/medcvr_ml/unity/unity_dataset.py
class UnityDataset():
    def __init__(self, root_dir, camera_type):
        if camera_type not in ['mono', 'stereo']:
            raise Exception("The camera can only be stereo or mono")
        self.root_dir = root_dir
        self.ann_def = AnnotationDefinitions(self.root_dir)
        self.keypoints = None
        self.keypoints = self._find_id('keypoints') if camera_type == 'stereo' else self._find_id('keypoints')[:1]
        self.catalog = Captures(self.root_dir)
        self.length = len(self.catalog.captures)

        # self.metric_def = MetricDefinitions(self.root_dir)
        # self.bbox2d_id = self._find_id('bounding box')
        # self.bbox3d_id = self._find_id('bounding box 3D')
        # self.segmap_id = self._find_id('semantic segmentation')

    def _find_id(self, name):
        # Find unique id using the name of the annotation, else keep empty
        mask = self.ann_def.table.name == name
        definition = self.ann_def.table[mask]

        if definition.empty:
            return None
        else:
            return definition['id'].tolist()

    def __len__(self):
        return self.length
    
    # adapted from Hanzi's code for get_keypoints_2D
    def get_keypoints(self, index):
        """Returns a List of Dicts of 2D keypoints"""
        if not self.keypoints:
            print('Unable to find keypoints')
            return None

        mask = (self.catalog.annotations['annotation_definition'] == self.keypoints[0])
        for i in range(1, len(self.keypoints)):
            mask = mask | (self.catalog.annotations['annotation_definition'] == self.keypoints[i])
        sub_df = self.catalog.annotations[mask]['values'].iloc[index]

        if len(sub_df) == 0:
            return
        result = list(filter(lambda point: not (point['x'] == 0 and point['y'] == 0), sub_df[0]['keypoints']))
        return result
    
    def get_img_filename(self, index):
        """Return filename of image frame of index given."""
        cell = self.catalog.captures["filename"].iloc[[index]]
        return cell[cell.index[0]]
    
    # def _get_filtered_catalog(self, id):
    #     if not id:
    #         return None

    #     return self.catalog.filter(id)

    # def get_image(self, index):
    #     path = os.path.join(self.root_dir, self.catalog.captures.iloc[index].filename)
    #     image = Image.open(path).convert("RGB")
    #     return image

    # def get_2d_bbox(self, index):
    #     """Returns a List of Dicts with 2D bounding boxes and labels"""
    #     if not self.bbox2d_id:
    #         print('This dataset does not have this annotation')
    #         return None

    #     mask = self.catalog.annotations['annotation_definition'] == self.bbox2d_id
    #     sub_df = self.catalog.annotations[mask].iloc[index]
    #     return sub_df['values']

    # def get_3d_bbox(self, index):
    #     """Returns a List of Dicts with 3D bounding boxes and labels"""
    #     if not self.bbox3d_id:
    #         print('This dataset does not have this annotation')
    #         return None

    #     mask = self.catalog.annotations['annotation_definition'] == self.bbox3d_id
    #     sub_df = self.catalog.annotations[mask].iloc[index]
    #     return sub_df['values']

    # def get_segmentation_map(self, index):
    #     """Returns the segmentation map image"""
    #     if not self.segmap_id:
    #         print('This dataset does not have this annotation')
    #         return None

    #     mask = self.catalog.annotations['annotation_definition'] == self.segmap_id
    #     sub_df = self.catalog.annotations[mask].iloc[index]
    #     image = Image.open(os.path.join(self.root_dir, sub_df['filename'])).convert("RGB")
    #     return image

def save_dataset_labels(unity_dataset_dir, keypoints, img_list, scorer, total_img_num, interleaving=True, save_dir=os.path.dirname(os.path.realpath('__file__'))):
    def construct_data_frame(keypoints, img_list, scorer):
        df = None
        placeholder = np.empty((len(img_list), 2))
        placeholder[:] = np.nan
        # img_list = np.sort(img_list)
        for keypoint in keypoints:
            cols = pd.MultiIndex.from_product(
                [[scorer], [keypoint], ['x', 'y']],
                names=['scorer', 'bodyparts', 'coords']
            )
            index = img_list
            frame = pd.DataFrame(placeholder, columns=cols, index=index)
            df = pd.concat([df, frame], axis=1)
        return df

    # read dataset
    is_mono = True
    has_left = False
    has_right = False
    for img_name in img_list:
        if "left" in img_name:
            has_left = True
        elif "right" in img_name:
            has_right = True
        if has_left and has_right:
            is_mono = False
            break
    camera_type = "mono" if is_mono else "stereo"
    dataset = UnityDataset(unity_dataset_dir, camera_type)
    # construct data frame
    df = construct_data_frame(keypoints, img_list, scorer)

    # real case: total_img_num - 2
    for i in range(min([len(dataset), total_img_num, len(img_list)])):
        # keypoint_coords = dataset.get_keypoints(int(img_list[i].split('.')[0].split('\\')[-1][3:]))
        keypoint_coords = dataset.get_keypoints(i)
        if not keypoint_coords:
            continue
        relative_filepath = dataset.get_img_filename(i)
        dir_separator = '/' if os.path.sep != '/' else os.path.sep
        relative_filepath = relative_filepath.split(dir_separator)[-1]
        row = df.loc[img_list[i]] # row data of the image with the filename

        j = 0
        while j < len(keypoint_coords):
            x = keypoint_coords[j]['x']
            y = keypoint_coords[j]['y']
            kp_name = keypoints[j]
            row[(scorer, kp_name, 'x')] = x
            row[(scorer, kp_name, 'y')] = y
            j += 1
    df.dropna(how='all', inplace=True)
    
    if not interleaving:
        df.sort_index(inplace=True)
        df = df.reindex(
            keypoints,
            axis=1,
            level=df.columns.names.index("bodyparts"),
        )

    # save dataframe as csv and h5 files
    df.to_csv(
        os.path.join(save_dir, "CollectedData_" + scorer + ".csv")
    )
    df.to_hdf(
        os.path.join(save_dir, "CollectedData_" + scorer + ".h5"),
        "df_with_missing",
    )
    return df



if __name__ == "__main__":
    unity_dataset_dir = os.path.abspath(r"C:\Users\peter\AppData\LocalLow\DefaultCompany\surgical_tool_tracking_research\inference\dataset\test\Dataset66b927eb-91d7-4016-8f3a-cdf949a187b7")
    dataset = UnityDataset(unity_dataset_dir)
