# Historical recovery

`before_layout_2026-11-04.zip` is the complete, byte-verified snapshot of all 359 physical files
under the original Midterm Project directory immediately before the four-folder migration.
It includes the old root README/editor settings, the November 1 deliverables, both earlier
archive ZIPs and manifests, and the prior midterm tree. Every original path, size and SHA-256 is
listed in `snapshot_manifest.json`. This single snapshot preserves the nested earlier history.

Extract into a NEW directory to inspect the historical project. Do not overwrite the current tree.
`records/` keeps small historical selection, cleanup and path-mapping records readily accessible.
Historical paths and upload flags describe their original run, not today's active selection.

The full snapshot ZIP is excluded from the normal code delivery to avoid recursively packaging
old code ZIPs. The extracted code delivery remains independently runnable without this snapshot.
