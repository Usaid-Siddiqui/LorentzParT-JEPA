#!/usr/bin/env bash
set -u
cd ~/demo/LorentzParT-JEPA
export PYTHONUNBUFFERED=1
mkdir -p experiments/phase2/diag_bestft
LOG=experiments/phase2/diag_bestft/overnight.log
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date) ==="

DATA=./data_1m
FT=configs/train_lorentz_part_1m.yaml
PRE=configs/pretrain_jepa_biased_1m.yaml
run(){ echo "--- $(date) :: $* ---"; "$@" || echo "[STAGE FAILED] $*"; }

# preserve the seed-123 result we already have
cp -n /tmp/bestft/seed123.json experiments/phase2/diag_bestft/ 2>/dev/null || true

# (1) SEED 456 — clean best-val finetune (prev run was killed at ep15)
echo "### SEED 456 best-val finetune ###"
rm -f logs/LorentzParT/best/jepa_1m_bestft_seed456.pt logs/LorentzParT/logging/jepa_1m_bestft_seed456.csv
run python scripts/train_lorentz_part.py --data-dir $DATA --config-path $FT \
    --run-name jepa_1m_bestft_seed456 --seed 456 \
    --weights logs/ParticleJEPA/best/jepa_1m_seed456_best.pt
run python scripts/evaluate.py --model lorentz_part --data-dir $DATA --config-path $FT \
    --weights logs/LorentzParT/best/jepa_1m_bestft_seed456.pt --seed 456 \
    --json-out experiments/phase2/diag_bestft/seed456.json

# (2) SEED 42 — re-pretrain (saves _best) then best-val finetune
echo "### SEED 42 pretrain + best-val finetune ###"
run python scripts/pretrain_jepa.py --data-dir $DATA --config-path $PRE \
    --run-name jepa_1m_seed42 --seed 42
if [ -f logs/ParticleJEPA/best/jepa_1m_seed42_best.pt ]; then
  run python scripts/train_lorentz_part.py --data-dir $DATA --config-path $FT \
      --run-name jepa_1m_bestft_seed42 --seed 42 \
      --weights logs/ParticleJEPA/best/jepa_1m_seed42_best.pt
  run python scripts/evaluate.py --model lorentz_part --data-dir $DATA --config-path $FT \
      --weights logs/LorentzParT/best/jepa_1m_bestft_seed42.pt --seed 42 \
      --json-out experiments/phase2/diag_bestft/seed42.json
else
  echo "[ERROR] seed42 _best not created — pretrain failed?"
fi

# SUMMARY
echo "=== SUMMARY $(date) ==="
python3 -c "
import json,os
ref={'42':(0.9152,0.9183),'123':(0.9153,0.9209),'456':(0.9173,0.9179)}
print('seed | best-val | ep30(final) | scratch')
for s in ('42','123','456'):
    p=f'experiments/phase2/diag_bestft/seed{s}.json'
    if os.path.exists(p):
        a=json.load(open(p))['roc_auc_ovo']; e,sc=ref[s]
        print(f'{s:>4} | {a:.4f}   | {e:.4f}      | {sc:.4f}')
    else:
        print(f'{s:>4} | (none)')
"
echo "=== DONE $(date) ==="
