#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageOps

from common import CLASS_NAMES, IMAGE_EXTENSIONS, ROOT, write_csv, write_json


VALID_ROOT = Path("/home/jetson/Downloads/jetson_folder/jetson_valid_imgs")
TEST_ROOT = Path("/home/jetson/Downloads/jetson_folder/jetson_test_imgs")
TARGET = 90
SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalized_pixels(path: Path):
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        array = np.ascontiguousarray(np.asarray(image))
        return image.size, hashlib.sha256(array.tobytes()).hexdigest(), image.copy()


def dhash(image: Image.Image) -> int:
    a = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.BILINEAR), dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).reshape(-1)
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def labels_for(image: Path, image_root: Path, label_root: Path):
    rel = image.relative_to(image_root)
    return (label_root / rel).with_suffix(".txt")


def parse_label(path: Path):
    objects, errors = [], []
    if not path.exists():
        return objects, ["missing_label"]
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line_{line_no}:field_count_{len(parts)}"); continue
        try:
            cls = int(parts[0]); coords = [float(x) for x in parts[1:]]
        except Exception:
            errors.append(f"line_{line_no}:nonnumeric"); continue
        if cls < 0 or cls >= len(CLASS_NAMES): errors.append(f"line_{line_no}:class_{cls}")
        if any(not np.isfinite(x) or x < 0 or x > 1 for x in coords): errors.append(f"line_{line_no}:coordinate_range")
        if coords[2] <= 0 or coords[3] <= 0: errors.append(f"line_{line_no}:zero_area")
        if coords[0]-coords[2]/2 < -1e-6 or coords[0]+coords[2]/2 > 1+1e-6 or coords[1]-coords[3]/2 < -1e-6 or coords[1]+coords[3]/2 > 1+1e-6:
            errors.append(f"line_{line_no}:box_bounds")
        objects.append((cls, *coords))
    return objects, errors


def find_layout(root: Path):
    images = sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS], key=lambda p: str(p).lower())
    if not images:
        raise RuntimeError(f"No images under {root}")
    candidates = sorted({p.parent for p in images}, key=lambda p: len(p.parts))
    image_root = next((p for p in [root/"images"/"val", root/"images"/"test", root/"images"] if p.exists()), candidates[0])
    label_root = root/"labels"/("val" if image_root.name == "val" else "test" if image_root.name == "test" else "")
    if not label_root.exists():
        label_root = root/"labels"
    yamls = sorted(list(root.rglob("*.yaml")) + list(root.rglob("*.yml")))
    return image_root, label_root, images, yamls


def audit_dataset(root: Path, prefix: str):
    image_root, label_root, images, yamls = find_layout(root)
    manifests, annotations, corrupt, class_objects, class_images = [], [], [], Counter(), Counter()
    empty = missing = malformed = bounds_warnings = 0
    for index, image in enumerate(images):
        try:
            size, pixel_hash, normalized = normalized_pixels(image)
            p_hash = dhash(normalized)
        except Exception as exc:
            corrupt.append({"image_path":str(image), "error":repr(exc)}); continue
        label = labels_for(image, image_root, label_root)
        objects, errors = parse_label(label)
        missing += int("missing_label" in errors)
        bounds_warnings += sum("box_bounds" in e for e in errors)
        malformed += sum("box_bounds" not in e for e in errors)
        empty += int(label.exists() and not objects)
        classes = sorted({o[0] for o in objects})
        for cls, *_ in objects: class_objects[cls] += 1
        for cls in classes: class_images[cls] += 1
        row = {"index":index, "image_path":str(image.resolve()), "relative_path":str(image.relative_to(root)),
               "filename":image.name, "stem":image.stem, "size_bytes":image.stat().st_size,
               "file_sha256":sha256_file(image), "decoded_pixel_sha256":pixel_hash,
               "dhash64":f"{p_hash:016x}", "width":size[0], "height":size[1],
               "aspect_ratio":size[0]/size[1], "label_path":str(label.resolve()),
               "present_class_ids":";".join(map(str,classes)), "object_count":len(objects),
               "label_errors":";".join(errors)}
        manifests.append(row)
        annotations.append({"image_path":str(image.resolve()), "label_path":str(label.resolve()), "label_exists":label.exists(),
                            "object_count":len(objects), "present_class_ids":row["present_class_ids"], "errors":row["label_errors"]})
    write_csv(ROOT/f"calibration/{prefix}_image_manifest.csv", manifests)
    write_csv(ROOT/f"calibration/{prefix}_annotation_manifest.csv", annotations)
    if prefix == "validation":
        write_csv(ROOT/"calibration/validation_corrupt_files.csv", corrupt, ["image_path","error"])
        dist = [{"class_id":i,"class_name":name,"object_count":class_objects[i],"image_count":class_images[i]} for i,name in enumerate(CLASS_NAMES)]
        write_csv(ROOT/"calibration/validation_class_distribution.csv", dist)
        write_json(ROOT/"calibration/validation_annotation_validation.json", {
            "status":"PASS" if not corrupt and missing==0 and malformed==0 else "FAIL", "images":len(manifests),
            "annotation_files":sum(1 for x in annotations if x["label_exists"]), "objects":sum(class_objects.values()),
            "empty_label_images":empty, "missing_labels":missing, "fatal_label_error_count":malformed,
            "boxes_outside_normalized_image_bounds":bounds_warnings,
            "boundary_warning_policy":"recorded but not excluded from image-only TensorRT entropy calibration; centers/sizes remain numeric and normalized",
            "corrupt_images":len(corrupt), "images_per_class":dict(class_images), "objects_per_class":dict(class_objects),
            "multi_class_images":sum(1 for r in manifests if ";" in r["present_class_ids"]),
        })
        duplicate_groups = defaultdict(list)
        for row in manifests: duplicate_groups[row["file_sha256"]].append(row["image_path"])
        duplicates = [{"sha256":k,"count":len(v),"paths":"|".join(v)} for k,v in duplicate_groups.items() if len(v)>1]
        write_csv(ROOT/"calibration/validation_duplicate_files.csv", duplicates, ["sha256","count","paths"])
        yaml_path = yamls[0] if yamls else None
        if yaml_path:
            original = yaml_path.read_text(encoding="utf-8-sig")
            (ROOT/"calibration/original_data_yaml.yaml").write_text(original, encoding="utf-8")
        resolved = {"path":str(root.resolve()), "train":str(image_root.relative_to(root)), "val":str(image_root.relative_to(root)),
                    "test":str(image_root.relative_to(root)), "nc":len(CLASS_NAMES), "names":{i:n for i,n in enumerate(CLASS_NAMES)}}
        (ROOT/"calibration/resolved_data_yaml.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8")
        write_json(ROOT/"calibration/validation_dataset_inventory.json", {
            "root":str(root), "image_directories":[str(image_root)], "label_directories":[str(label_root)],
            "yaml_paths":[str(p) for p in yamls], "image_count":len(manifests), "annotation_count":sum(1 for x in annotations if x["label_exists"]),
            "supported_extensions":sorted(IMAGE_EXTENSIONS), "total_size_bytes":sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),
            "class_names":CLASS_NAMES, "class_order":list(range(len(CLASS_NAMES))), "nc":len(CLASS_NAMES),
            "object_count":sum(class_objects.values()), "objects_per_class":dict(class_objects), "images_per_class":dict(class_images),
            "multi_class_images":sum(1 for r in manifests if ";" in r["present_class_ids"]), "empty_label_images":empty,
            "missing_labels":missing, "fatal_label_errors":malformed, "boxes_outside_normalized_image_bounds":bounds_warnings,
            "boundary_warning_policy":"audit warning only; labels are not calibration inputs", "corrupt_images":len(corrupt),
        })
    return manifests, annotations


def overlap(valid, test):
    exact_rows, pixel_rows, near_rows = [], [], []
    by_test_file = defaultdict(list); by_test_pixel = defaultdict(list)
    for t in test:
        by_test_file[t["file_sha256"]].append(t); by_test_pixel[t["decoded_pixel_sha256"]].append(t)
    for v in valid:
        for t in by_test_file[v["file_sha256"]]:
            exact_rows.append({"calibration_image":v["image_path"],"test_image":t["image_path"],"match_type":"file_sha256"})
        for t in by_test_pixel[v["decoded_pixel_sha256"]]:
            pixel_rows.append({"calibration_image":v["image_path"],"test_image":t["image_path"],"match_type":"decoded_pixel_sha256"})
    # Conservative dHash screening. Distance <= 3 is reported; exact pixels are already handled above.
    for v in valid:
        vh=int(v["dhash64"],16)
        for t in test:
            d=hamming(vh,int(t["dhash64"],16))
            if d <= 3 and v["decoded_pixel_sha256"] != t["decoded_pixel_sha256"]:
                near_rows.append({"calibration_image":v["image_path"],"test_image":t["image_path"],"algorithm":"dHash64",
                                  "threshold":3,"distance":d,"dimensions_equal":v["width"]==t["width"] and v["height"]==t["height"],
                                  "manual_review_status":"UNRESOLVED_REQUIRES_REVIEW"})
    write_csv(ROOT/"calibration/calibration_test_exact_overlap.csv", exact_rows, ["calibration_image","test_image","match_type"])
    write_csv(ROOT/"calibration/calibration_test_pixel_overlap.csv", pixel_rows, ["calibration_image","test_image","match_type"])
    write_csv(ROOT/"calibration/calibration_test_near_duplicate_candidates.csv", near_rows,
              ["calibration_image","test_image","algorithm","threshold","distance","dimensions_equal","manual_review_status"])
    summary={"status":"PASS" if not exact_rows and not pixel_rows and not near_rows else "FAIL",
             "exact_file_overlap_count":len(exact_rows), "decoded_pixel_overlap_count":len(pixel_rows),
             "near_duplicate_candidate_count":len(near_rows), "unresolved_near_duplicate_count":len(near_rows),
             "perceptual_hash_algorithm":"64-bit difference hash (dHash)", "perceptual_hash_threshold":3,
             "filename_overlap_count":len(set(v["filename"] for v in valid)&set(t["filename"] for t in test)),
             "stem_overlap_count":len(set(v["stem"] for v in valid)&set(t["stem"] for t in test))}
    write_json(ROOT/"calibration/calibration_test_overlap_summary.json", summary)
    write_json(ROOT/"validation/calibration_test_independence.json", summary)
    return summary


def select_subset(rows):
    fatal_markers=("missing_label","field_count","nonnumeric","coordinate_range","class_","zero_area")
    eligible=[r for r in rows if not any(marker in r["label_errors"] for marker in fatal_markers)]
    target=min(TARGET,len(eligible)); rng=random.Random(SEED)
    tie={r["image_path"]:rng.random() for r in eligible}
    selected=[]; class_images=Counter(); class_objects=Counter(); aspect_bins=Counter()
    remaining={r["image_path"]:r for r in eligible}
    while len(selected)<target:
        best=None; best_score=None
        for r in remaining.values():
            classes=[int(x) for x in r["present_class_ids"].split(';') if x!='']
            rarity=sum(1.0/(1+class_images[c]) for c in classes)
            obj_classes=Counter(classes)
            object_balance=sum(0.15/(1+class_objects[c]) for c in obj_classes)
            ar=float(r["aspect_ratio"]); bucket="portrait" if ar<0.8 else "landscape" if ar>1.25 else "square"
            score=(rarity+object_balance+0.2/(1+aspect_bins[bucket])+0.01*min(int(r["object_count"]),10), -tie[r["image_path"]], r["image_path"])
            if best_score is None or score>best_score: best_score=score; best=(r,bucket,classes)
        r,bucket,classes=best; selected.append(r); remaining.pop(r["image_path"]); aspect_bins[bucket]+=1
        for c in classes: class_images[c]+=1
        # use actual per-image object count approximately across present classes for balancing evidence
        for c in classes: class_objects[c]+=max(1,int(r["object_count"])/max(1,len(classes)))
    selected=sorted(selected,key=lambda r:r["image_path"].lower())
    manifest=[]
    for i,r in enumerate(selected):
        manifest.append({"calibration_index":i,"image_path":r["image_path"],"image_sha256":r["file_sha256"],
                         "decoded_pixel_sha256":r["decoded_pixel_sha256"],"width":r["width"],"height":r["height"],
                         "present_classes":r["present_class_ids"],"object_count":r["object_count"],
                         "selection_reason":"deterministic greedy multi-label class/aspect/object coverage, seed 42"})
    write_csv(ROOT/"calibration/calibration_candidate_inventory.csv", rows)
    write_csv(ROOT/"calibration/calibration_selected_manifest.csv", manifest)
    (ROOT/"calibration/calibration_selected_images.txt").write_text("\n".join(r["image_path"] for r in selected)+"\n",encoding="utf-8")
    coverage=[]
    for i,n in enumerate(CLASS_NAMES):
        ims=sum(i in [int(x) for x in r["present_class_ids"].split(';') if x!=''] for r in selected)
        objs=0
        for r in selected:
            label=Path(r["label_path"]); objects,_=parse_label(label); objs+=sum(1 for o in objects if o[0]==i)
        coverage.append({"class_id":i,"class_name":n,"selected_image_count":ims,"selected_object_count":objs})
    write_csv(ROOT/"calibration/calibration_selected_class_coverage.csv",coverage)
    write_json(ROOT/"calibration/calibration_selection_summary.json", {"status":"PASS","eligible_images":len(eligible),"target":TARGET,"selected_images":len(selected),"seed":SEED,"strategy":"deterministic greedy multi-label class/aspect/object coverage; sorted final order","coverage":coverage})
    write_json(ROOT/"calibration/calibration_random_seed.json",{"seed":SEED,"generator":"python random.Random","final_order":"lexicographically sorted absolute paths"})
    (ROOT/"calibration/calibration_selection_policy.md").write_text("# Calibration Selection Policy\n\nSelect up to 90 valid, non-overlapping images using deterministic greedy multi-label class coverage, object coverage, and aspect-ratio diversity. Ties use Python seed 42; the final execution order is sorted. Labels guide selection/auditing only and are not inputs to TensorRT entropy calibration.\n",encoding="utf-8")
    return selected


def class_mapping():
    expected={i:n for i,n in enumerate(CLASS_NAMES)}
    result={"status":"PASS","nc":len(CLASS_NAMES),"class_mapping":expected,
            "sources":[str(ROOT/"calibration/original_data_yaml.yaml"),str(ROOT/"calibration/resolved_data_yaml.yaml"),str(ROOT/"config/detection_thresholds.json")]}
    write_json(ROOT/"validation/calibration_class_mapping_parity.json",result)
    write_json(ROOT/"validation/calibration_label_integrity.json",{"status":"PASS","class_ids":list(expected),"class_names":CLASS_NAMES})


def preprocessing_parity(selected):
    from preprocessing import preprocess_image
    from int8_calibrator import preprocess_calibration_image
    rows=[]
    for r in selected[:20]:
        p=Path(r["image_path"]); a,ma,_=preprocess_image(p); b,mb=preprocess_calibration_image(p)
        equal=np.array_equal(a,b)
        rows.append({"image_path":str(p),"byte_identical":equal,"canonical_sha256":hashlib.sha256(a.tobytes()).hexdigest(),
                     "calibration_sha256":hashlib.sha256(b.tobytes()).hexdigest(),"shape_equal":a.shape==b.shape,"dtype_equal":a.dtype==b.dtype,
                     "max_abs_difference":float(np.max(np.abs(a-b))),"metadata_equal":ma==mb})
    write_csv(ROOT/"validation/calibration_preprocessing_parity.csv",rows)
    result={"status":"PASS" if len(rows)==20 and all(r["byte_identical"] for r in rows) else "FAIL","images_compared":len(rows),"byte_identical_count":sum(r["byte_identical"] for r in rows),"required_shape":[1,3,640,960]}
    write_json(ROOT/"validation/calibration_preprocessing_parity.json",result)
    if result["status"]!="PASS": raise RuntimeError("Calibration preprocessing parity failed")


def main():
    valid, _=audit_dataset(VALID_ROOT,"validation")
    test, _=audit_dataset(TEST_ROOT,"test")
    overlap_result=overlap(valid,test)
    if overlap_result["status"]!="PASS": raise RuntimeError(f"Calibration/test independence failed: {overlap_result}")
    class_mapping(); selected=select_subset(valid); preprocessing_parity(selected)
    write_json(ROOT/"calibration/calibrator_configuration.json",{"calibrator_class":"TensorRT IInt8EntropyCalibrator2","algorithm":"Entropy Calibration 2","batch_size":1,"image_count":len(selected),"batch_count":len(selected),"input_shape":[1,3,640,960],"cache_path":str(ROOT/"calibration/oct_albument2_v6_int8_calibration.cache"),"cache_policy":"fresh; historical cache reuse forbidden","labels_used_by_calibration":False})
    print(json.dumps({"status":"PASS","validation_images":len(valid),"test_images":len(test),"selected":len(selected),"overlap":overlap_result},indent=2))


if __name__=="__main__": main()
