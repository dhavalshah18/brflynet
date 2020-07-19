import pathlib
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image


class MRAProjected(data.Dataset):
    """
    Class defined to handle 3D MRA data reprojected as
    2D axial and saggital slices

    MIP images should already be created and in folder
    split into train and val with txt file

    raw_dir
        - /case_dir
            - /pre/TOF.nii.gz

    img_dir
        -/raw
            - /case_dir_axial*.png
            - /case_dir_sag*.png
        -/seg
            - /case_dir_axial*_label.png
            - /case_dir_sag*_label.png
    """

    def __init__(self, cfg, transform=None, mode="train"):
        super().__init__()
        self.cfg = cfg
        self.orig_dir = pathlib.Path(cfg.ORIGINAL_PATH)
        self.img_dir = pathlib.Path(cfg.MIP_PATH)
        # Dictionary storing paths of all axial and sagittal images to use for dataset
        self.img_list = {"axial": [], "sag": []}
        self.transform = transform
        self.resize = transforms.Resize((256, 256))
        self.mode = mode

        # Set the split file
        if self.mode == "train":
            self.split_file = pathlib.Path(cfg.TRAIN_SPLIT_FILE)
        elif self.mode == "val":
            self.split_file = pathlib.Path(cfg.VAL_SPLIT_FILE)
        elif self.mode == "":
            self.split_file = None
        else:
            raise Exception("Mode can only be train or val or empty")

        # Read paths of images into img_list from split file
        if self.split_file:
            with open(str(self.split_file), "r") as sf:
                case_nums = sf.readlines()
            self.get_img_list(case_nums)
            self.raw_case_dirs = [self.orig_dir.joinpath(c.strip() + "/") for c in case_nums]
        # If no split file, just use all image files
        else:
            self.get_img_list()
            self.raw_case_dirs = [d for d in sorted(self.orig_dir.glob("100*/"))]

    def __len__(self):
        return len(self.img_list["axial"])

    def __getitem__(self, index):
        axial_path = self.img_list["axial"][index]
        sag_path = self.img_list["sag"][index]

        axial_label_path = str(axial_path)[:-4].replace("raw", "seg") + "_label.png"
        sag_label_path = str(sag_path)[:-4].replace("raw", "seg") + "_label.png"

        axial_img = self.resize(Image.open(str(axial_path)))
        axial_img = transforms.ToTensor()(axial_img)
        sag_img = self.resize(Image.open(str(sag_path)))
        sag_img = transforms.ToTensor()(sag_img)

        # TODO size of axial and sag img should be the same???

        axial_label = self.resize(Image.open(str(axial_label_path)))
        axial_label = transforms.ToTensor()(axial_label)[0]
        print(torch.unique(axial_label))
        sag_label = self.resize(Image.open(str(sag_label_path)))
        sag_label = transforms.ToTensor()(sag_label)[0]
        print(torch.unique(sag_label))

        # TODO size of axial and sag label should be the same

        img_name = str(axial_path).split("/")[-1]
        case_num = img_name[0:5]

        sample = {"top_image": axial_img, "bottom_image": sag_img, "top_label": axial_label, "bottom_label": sag_label,
                  "case_num": case_num}

        return sample

    def get_img_list(self, case_num=None):
        """
        Get paths of all images given a case number.
        If no case number, return all image paths in img_dirs folder
        :param case_num:
        """
        if not case_num:
            axial = list(sorted(self.img_dir.glob("raw/*_axial*.png")))
            sag = list(sorted(self.img_dir.glob("raw/*_sag*.png")))

            assert len(axial) == len(sag), "Number of axial and saggital images should be the same"

            self.img_list["axial"].append(axial)
            self.img_list["sag"].append(sag)

        elif isinstance(case_num, str):
            ax_name = case_num + "_axial*.png"
            sag_name = case_num + "_sag*.png"
            axial = list(sorted(self.img_dir.glob("raw/" + ax_name)))
            sag = list(sorted(self.img_dir.glob("raw/" + sag_name)))

            assert len(axial) == len(sag), "Number of axial and saggital images are not the same for case {0}".format(case_num)

            ax_list = self.img_list.get("axial", [])
            sag_list = self.img_list.get("sag", [])

            ax_list.append([i for i in axial])
            sag_list.append([i for i in sag])

            self.img_list["axial"] = ax_list
            self.img_list["sag"] = sag_list

        elif isinstance(case_num, list):
            for num in case_num:
                num = num.strip()
                self.get_img_list(num)
                
            self.img_list["axial"] = [item for sublist in self.img_list["axial"] for item in sublist]
            self.img_list["sag"] = [item for sublist in self.img_list["sag"] for item in sublist]


        else:
            raise Exception("Unrecognized format for case_num input")

