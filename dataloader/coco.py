import os
import json
from pycocotools.coco import  COCO as Coco


class COCO:
    def __init__(self, folder):
        self.annotations = {}
        self.folder = folder

    def organize(self, cap_filename, det_filename):
        # assert cap_filename is not None and det_filename is not None, "Please provide both instance and caption path."
        self.load_caption(cap_filename)
        self.load_bb(det_filename)
        # return self.annotations

    def load_caption(self, filename):
        file = os.path.join(self.folder, filename)
        coco_ann = Coco(file)
        ids = list(coco_ann.anns.keys())
        for id in ids:
            image_id = coco_ann.anns[id]["image_id"]
            img_filename = coco_ann.loadImgs(image_id)[0]["file_name"]
            caption1 = coco_ann.anns[id]["caption"]
            self.annotations[img_filename] = {}
            self.annotations[img_filename]["image_id"] = image_id
            self.annotations[img_filename]["caption"] = []
            self.annotations[img_filename]["caption"].append(caption1)

            # load the other remaining 4.
            ids_ann = coco_ann.getAnnIds(imgIds=image_id)
            # Filter the indexes,
            filtered_id = []
            for cap_id in ids_ann:
                if cap_id != id:
                    filtered_id.append(cap_id)
            filtered_id = filtered_id[:4]

            for cap_id in filtered_id:
                caption = coco_ann.anns[cap_id]["caption"]
                self.annotations[img_filename]["caption"].append(caption)

            assert len(self.annotations[img_filename]["caption"]) == 5, f"Length of each image caption should be five," \
                                                                       f"but get {len(self.annotations[img_filename]['caption'])}" \
                                                                        f" for image_id: {image_id}."

        return self.annotations

    def load_bb(self, filename):
        file = os.path.join(self.folder, filename)
        instance_ann = Coco(file)
        ids = list(instance_ann.anns.keys())

        for id in ids:
            image_id = instance_ann.anns[id]["image_id"]
            all_id = instance_ann.getAnnIds(image_id)
            img_name = instance_ann.loadImgs(image_id)[0]["file_name"]
            bbox = []
            cat = []
            for i in all_id:
                bbox.append(instance_ann.anns[i]["bbox"])
                cat.append(instance_ann.anns[i]["category_id"])

            self.annotations[img_name]["bbox"] = bbox
            self.annotations[img_name]["category_id"] = cat

        return self.annotations

    def save(self, dictionary, filepath):
        with open(filepath, 'w') as f:
            json.dump(dictionary, f)
            print(f"succefully saved to {filepath}.")

    def load(self, filepath):
        with open(filepath, 'r') as f:
            print("loading ...")
            file = json.load(f)
        return file


if __name__ == '__main__':
    ann_folder = ".../coco/annotation/annotations/"
    train_cap = "captions_train2017.json"
    train_ins = "instances_train2017.json"
    val_cap = "captions_val2017.json"
    val_ins = "instances_val2017.json"
    tsave_to = "cap_bbox_train2017.json"
    vsave_to = "cap_bbox_val2017.json"

    coco = COCO(ann_folder)
    coco.organize(cap_filename=train_cap,
                  det_filename=train_ins)

    if not os.path.exists(tsave_to):
        coco.save(coco.annotations, tsave_to)

    coco = COCO(ann_folder)
    coco.organize(cap_filename=val_cap,
                  det_filename=val_ins)

    if not os.path.exists(vsave_to):
        coco.save(coco.annotations, vsave_to)















