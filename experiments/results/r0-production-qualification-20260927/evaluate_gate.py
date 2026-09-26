"""One-off frozen R0 D_gate evaluation; no training or cloud writes."""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import pandas as pd
import torch
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from llm_router_labeling.evaluation.continual_gate import evaluate_continual_predictions
from llm_router_labeling.evaluation.routerarena_routing_evaluator import LoadedHuggingFaceRouterPredictor

root = Path(__file__).resolve().parent
repo = root.parents[2]
qualification = root / 'qualification'
rules_path = qualification / 'initial-rules.json'
pilot_path = qualification / 'initial-pilot.json'
assert rules_path.exists() and pilot_path.exists(), 'Freeze D_dev rules before D_gate'
sha = lambda p: hashlib.file_digest(p.open('rb'), 'sha256').hexdigest()
rules = json.loads(rules_path.read_text())
assert rules['threshold_selection']['report_sha256'] == sha(pilot_path)
out = root / 'gate'
out.mkdir(exist_ok=False)
config = json.loads((repo / '.worktrees/r0-ml/ml/config/r0_continual_gate_8192.json').read_text())
dataset = repo / 'ml/artifacts/modernbert-initial-v1/dataset/sealed/gate.parquet'
assert sha(dataset) == config['evaluation_dataset_sha256']
model_path = root / 'source/extracted/final-model'
weights_sha = sha(model_path / 'model.safetensors')
assert weights_sha == json.loads(pilot_path.read_text())['candidate_weights_sha256']
receipt = {'started_at_unix': time.time(), 'rules_sha256': sha(rules_path), 'pilot_sha256': sha(pilot_path), 'weights_sha256': weights_sha, 'dataset_sha256': sha(dataset), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': 'cpu', 'attention': 'eager', 'dtype': 'float32', 'training_performed': False}
(out / 'start.json').write_text(json.dumps(receipt, indent=2))
torch.set_num_threads(4)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True, attn_implementation='eager', torch_dtype=torch.float32).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
predictor = LoadedHuggingFaceRouterPredictor(model, tokenizer, torch, checkpoint_path=str(model_path), label_order=config['label_order'], batch_size=1, max_length=8192, device='cpu')
frame = pd.read_parquet(dataset)
parts = []
started = time.monotonic()
for offset in range(0, len(frame), 64):
    part = predictor.predict(frame.iloc[offset:offset+64]['model_input_text'].astype(str).tolist())
    part.to_parquet(out / f'part-{offset:05d}.parquet', index=False)
    parts.append(part)
    print(json.dumps({'completed': min(offset+64,len(frame)), 'total': len(frame), 'seconds': round(time.monotonic()-started,1)}), flush=True)
info = dict(predictor.info())
info.update({'model_ownership': 'frozen-checkpoint-evaluation', 'weights_sha256': weights_sha, 'torch': torch.__version__, 'transformers': transformers.__version__, 'attention': 'eager', 'dtype': 'float32'})
result = evaluate_continual_predictions(frame, pd.concat(parts, ignore_index=True), config, model_info=info)
result.predictions.to_parquet(out / 'predictions.parquet', index=False)
(out / 'metrics.json').write_text(json.dumps(result.metrics, indent=2, sort_keys=True))
receipt.update({'completed_at_unix': time.time(), 'metrics_sha256': sha(out/'metrics.json'), 'predictions_sha256':sha(out/'predictions.parquet')})
(out / 'receipt.json').write_text(json.dumps(receipt, indent=2))
print('Completed frozen D_gate evaluation', flush=True)
