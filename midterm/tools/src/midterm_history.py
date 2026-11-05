"""Resolve frozen evidence after an authorized update to current final outputs."""
from pathlib import Path
import json
from midterm_evaluation import file_sha256


def history_manifest(root):
    path=Path(root)/'history/manifest.json'
    return json.loads(path.read_text()) if path.exists() else {'archived':{},'regenerated':{}}


def historical_path(root, relative):
    root=Path(root);relative=str(relative)
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('History paths must remain inside the project.')
    record=history_manifest(root)['archived'].get(relative)
    if record is None:return root/relative
    target=root/record['archive_path']
    if not str(target.resolve()).startswith(str((root/'history').resolve())+'/'):
        raise ValueError('Invalid archive location.')
    if file_sha256(target)!=record['sha256']:
        raise AssertionError(f'Archived evidence changed: {relative}')
    return target


def generated_archive_exception(root, relative, digest):
    """Only explicitly recorded prior delivery ZIPs may omit archived bytes."""
    if str(relative) != 'deliverables/midterm_code.zip':
        return False
    record=history_manifest(root).get('regenerated',{}).get(str(relative),{})
    return digest in [record.get('before_sha256'),*record.get('known_prior_sha256',[])]


def resolve_expected_path(root, relative, digest):
    """Resolve an exact historical digest, retaining multiple prior versions."""
    root=Path(root);relative=str(relative);path=Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('History paths must remain inside the project.')
    current=root/path
    if current.is_file() and file_sha256(current)==digest:
        return current
    manifest=history_manifest(root)
    record=manifest.get('archived',{}).get(relative,{})
    archive_path=record.get('archive_path') if record.get('sha256')==digest else None
    archive_path=manifest.get('versions',{}).get(relative,{}).get(digest,archive_path)
    if archive_path:
        target=root/archive_path
        if not target.resolve().is_relative_to((root/'history').resolve()):
            raise ValueError('Invalid archive location.')
        if not target.is_file() or file_sha256(target)!=digest:
            raise AssertionError(f'Archived evidence changed: {relative}')
        return target
    if not current.is_file():
        raise FileNotFoundError(current)
    raise AssertionError(f'Original file changed without an exact archived version: {relative}')


def verify_snapshot(root, snapshot):
    """Verify original bytes in place or in the declared pre-integration archive."""
    root=Path(root);manifest=history_manifest(root)
    counts={'verified_current':0,'verified_archived':0,'excluded_regenerated_archive':0}
    for relative,digest in snapshot.items():
        if generated_archive_exception(root,relative,digest):
            counts['excluded_regenerated_archive']+=1
            continue
        path=resolve_expected_path(root,relative,digest)
        if file_sha256(path)!=digest:raise AssertionError(f'Original file changed: {relative}')
        counts['verified_current' if path==root/relative else 'verified_archived']+=1
    return {'status':'PASS','snapshot_entries':len(snapshot),**counts}


def baseline_model_path(root):
    return Path(root)/'models/part2/01_tfidf_lr/model.joblib'


def augmentation_metadata_compatible(saved,current):
    """Separate code provenance and CPU discovery warnings from fitted-result identity.

    Callers additionally require identical parameters, saved labels, predictions,
    and artifact checksums. Solver warnings and all other metadata remain strict.
    """
    import math
    left=dict(saved);right=dict(current)
    provenance=['training_code_sha256','augmentation_module_sha256']
    for key in provenance:
        left.pop(key,None);right.pop(key,None)
    for metadata in [left,right]:
        for key in ['cluster_warnings','classifier_warnings']:
            metadata[key]=[warning for warning in metadata.get(key,[])
                if not warning.startswith('Could not find the number of physical cores for the following reason:')]
    key='baseline_control_max_probability_difference'
    a=left.pop(key,None);b=right.pop(key,None)
    numeric=(isinstance(a,(float,int)) and isinstance(b,(float,int)) and
             math.isfinite(a) and math.isfinite(b) and max(abs(a),abs(b))<=1e-10 and abs(a-b)<=1e-12)
    return numeric and left==right
