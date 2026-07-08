# pointcloud-viewer

Browser viewer for mirage's 3D anomaly detection — rotate a scan as a colored point cloud, toggle
between **rgb**, the **anomaly heatmap** (the working PatchCore detector's score), and the **GT defect
mask**. The cardioview analog: *see the model work*, in-browser, shareable.

Self-contained — one `index.html` (three.js via CDN, no build step). Data is **MVTec-derived
(CC BY-NC-SA)** so `data/` is gitignored; generate it locally.

## Run
```bash
# 1. generate sample data (each fits a PatchCore bank + scores the sample):
python -m surfscan.viz export --samples \
    bagel:test:hole:0 bagel:train:good:0 cookie:test:crack:0 tire:test:cut:0 carrot:test:hole:0
# -> pointcloud-viewer/data/<id>.json + manifest.json

# 2. serve (fetch needs http, not file://) and open:
cd pointcloud-viewer && python -m http.server 8000
# then open http://localhost:8000
```

Pick a sample (dropdown), switch **color by** → on a defect sample, the **anomaly** mode should light up
the defect (matching the **GT** mode) — the detector localizing, made visible.

## Note
`--samples` is `cat:split:defect:idx`. The exported anomaly is the PatchCore (rgb feature memory bank)
score, normalized 0–1. For a public live demo (GitHub Pages), redistribution of MVTec-derived points
falls under the dataset's non-commercial license — attribute MVTec and check terms first.
