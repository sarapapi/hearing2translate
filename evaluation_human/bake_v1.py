# clip only to annotated and merge them into a single campaign
# the original annotations were done on an older version of Pearmut, which is why we need to do this step

# %%

import json
import os
import shutil

with open("hearing2translate-v1/annotations.json", "r") as f:
    data_annotations = json.load(f)["hearing2translate-v1"]

data_campaign_new = {
    "campaign_id":"hearing2translate-v1",
    "info": {
        "assignment":"task-based",
        "protocol":"MQM",
        "assignment": "single-stream",
        "users": ["user"]
    },
    "data": []
}

data_annotations_new = []
data_progress_new = {"user": {"progress": []}}

for document in data_annotations:
    doc_item_new = {
        "item": [],
        "annotation": [],
        "item_i": None,
    }
    assets = []
    for item_annotation, item_item in zip(document["annotations"], document["item"]):
        new_item_item = {"src": item_item["src"], "tgt": {}}
        assets.append(item_item["src"].split(' src="')[1].split('"></audio>')[0])
        new_item_annotation = {}
        for model, tgt, annotation in zip(item_item["models"], item_item["tgt"], item_annotation):
            new_item_item["tgt"][model] = tgt
            new_item_annotation[model] = annotation
        doc_item_new["item"].append(new_item_item)
        doc_item_new["annotation"].append(new_item_annotation)


    # skip no error spans
    if all(not annotation_obj["error_spans"] for annotation in doc_item_new["annotation"] for annotation_obj in annotation.values()):
        continue

    # copy assets
    for asset in assets:
        fname0 = f"data/{asset}"
        fname1 = f"/home/vilda/Downloads/hearing2translate-v1-baked/{asset}"
        os.makedirs(os.path.dirname(fname1), exist_ok=True)
        if not os.path.exists(fname1):
            shutil.copy(fname0, fname1)

    data_progress_new["user"]["progress"].append("completed")
    data_campaign_new["data"].append(doc_item_new["item"])
    doc_item_new["item_i"] = len(data_campaign_new["data"]) - 1
    data_annotations_new.append(doc_item_new)

with open("/home/vilda/Downloads/campaigns_fixed.json", "w") as f:
    json.dump([data_campaign_new], f)

with open("/home/vilda/Downloads/annotations_fixed.json", "w") as f:
    json.dump({"hearing2translate-v1": data_annotations_new}, f)

with open("/home/vilda/Downloads/progress_fixed.json", "w") as f:
    json.dump({"hearing2translate-v1": data_progress_new}, f)

"""
cd evaluation_human
pearmut bake-existing --campaigns ~/Downloads/campaigns_fixed.json --progress ~/Downloads/progress_fixed.json --annotations ~/Downloads/annotations_fixed.json --output ~/Downloads/hearing2translate-v1-baked
"""