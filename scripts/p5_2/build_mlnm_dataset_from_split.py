#!/usr/bin/env python3
"""Build an MLNM dataset from a frozen patient-level split, excluding approved patches."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from PIL import Image
from p5_2.build_mlnm_dataset import LABELS,build,load_records,scan_patients,write_csv

def main():
 p=argparse.ArgumentParser()
 for n in ("cleaning-csv","split-csv","output-root","run-dir"): p.add_argument("--"+n,type=Path,required=True)
 p.add_argument("--exclude-item-id",action="append",default=[])
 a=p.parse_args()
 if a.output_root.exists(): raise FileExistsError(a.output_root)
 rows=load_records(a.cleaning_csv)
 with a.split_csv.open(encoding="utf-8",newline="") as f: sr=list(csv.DictReader(f))
 assignment={r["patient"]:r["split"] for r in sr}
 if len(assignment)!=99 or len(sr)!=99 or set(assignment.values())!={"train","val","test"}: raise ValueError("invalid patient split manifest")
 excluded=set(a.exclude_item_id)
 byid={f"{r['patient']}__{Path(r['image_path']).stem}":r for r in rows}
 if len(byid)!=len(rows) or not excluded<=byid.keys(): raise ValueError("unknown or duplicate exclusion")
 for item in excluded:
  if not np.any(np.asarray(Image.open(byid[item]["cleaned_label_path"]))==9): raise ValueError(f"class 9 absent: {item}")
 kept=[r for item,r in byid.items() if item not in excluded]
 patients,_=scan_patients(kept)
 if set(patients)!=set(assignment): raise ValueError("patient mismatch")
 groups={s:sorted(p for p,v in assignment.items() if v==s) for s in ("train","val","test")}
 manifest=build(kept,groups,a.output_root)
 a.output_root.mkdir(parents=True,exist_ok=True)
 write_csv(a.output_root/"manifest.csv",manifest,list(manifest[0]))
 write_csv(a.output_root/"split.csv",sorted(({"patient":p,"split":s} for p,s in assignment.items()),key=lambda r:r["patient"]),["patient","split"])
 write_csv(a.output_root/"exclusions.csv",[{"item_id":i,"reason":"approved class-9 exclusion; source preserved"} for i in sorted(excluded)],["item_id","reason"])
 (a.output_root/"class_map.json").write_text(json.dumps(LABELS,indent=2)+"\n")
 pixels=Counter(); links=True
 for r in manifest:
  v,c=np.unique(np.asarray(Image.open(r["label"])),return_counts=True); pixels.update({int(x):int(y) for x,y in zip(v,c)})
  links &= Path(r["image"]).stat().st_ino==Path(r["source_image"]).stat().st_ino and Path(r["label"]).stat().st_ino==Path(r["source_label"]).stat().st_ino
 problems=[]
 if len(manifest)!=len(rows)-len(excluded): problems.append("patch count mismatch")
 if pixels[9]: problems.append("class 9 remains in output")
 if not links: problems.append("hard-link inode mismatch")
 val={"patch_count":len(manifest),"patient_count":len(assignment),"patient_split_counts":{s:len(v) for s,v in groups.items()},"patch_split_counts":dict(Counter(r["split"] for r in manifest)),"class_pixel_counts":dict(sorted(pixels.items())),"excluded_item_ids":sorted(excluded),"class_9_pixels":pixels[9],"hardlinks_validated":links,"problems":problems,"split_sha256":hashlib.sha256((a.output_root/"split.csv").read_bytes()).hexdigest(),"manifest_sha256":hashlib.sha256((a.output_root/"manifest.csv").read_bytes()).hexdigest()}
 (a.output_root/"validation.json").write_text(json.dumps(val,indent=2)+"\n")
 (a.output_root/"README.md").write_text("# MLNM dataset v2\n\nPatient-level split.csv is authoritative; manifest split values are derived. Excluded source patches remain untouched.\n",encoding="utf-8")
 a.run_dir.mkdir(parents=True,exist_ok=True)
 (a.run_dir/"summary.json").write_text(json.dumps({"dataset_root":str(a.output_root),**val},indent=2)+"\n")
 if problems: raise RuntimeError(problems)
if __name__=="__main__": main()
