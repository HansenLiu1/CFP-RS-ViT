# Manifests

The pipeline creates the following per-run manifest files:

```text
manifest/
  run.json
  environment.txt
  source_files.csv
  images.csv
  objects.csv
  splits.csv
  models.csv
```

`source_files.csv` records the dataset path, file identity and SHA-256 of every
raw image and label.  `images.csv` and `objects.csv` record the parsed geometry,
class and exclusion status.  `splits.csv` assigns source images to the four
roles with no source-image overlap.  `models.csv` records checkpoint and loader
identity.

These files are generated, not committed.  A compact summary may be placed in
`artifacts/paper/` if it is needed to audit a reported number.
