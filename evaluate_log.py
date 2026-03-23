import re

# ===== CHANGE THIS =====
LOG_PATH = "/home/zhenrant/rendering_prompted_muggled_sam/train_logs/0321_multi_gt_k12_bbox_b156_resume_from_singlegte8_s0.log"
# LOG_PATH = "/home/kevin/muggled_sam/train_logs/0315_finetune_new_multi_gt_dataset_k15_b154_from_e21.log"

# =======================


PQ_RE = re.compile(
    r"\[eval\]\s+PQ@score>=([0-9.]+)=([0-9.]+)\s+tp=(\d+)\s+fp=(\d+)\s+fn=(\d+)"
)

FINAL_EVAL_RE = re.compile(
    r"\[eval\]\s+epoch=(\d+)\s+avg_iou=([0-9.]+)\s+correct_rate=([0-9.]+)\s+samples=(\d+)"
)


results = []
current_pq = []

with open(LOG_PATH, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        line = line.strip()

        pq_match = PQ_RE.match(line)
        if pq_match:
            threshold = float(pq_match.group(1))
            pq_score = float(pq_match.group(2))

            current_pq.append((threshold, pq_score))
            continue

        final_match = FINAL_EVAL_RE.match(line)
        if final_match:
            eval_epoch = int(final_match.group(1))
            avg_iou = float(final_match.group(2))
            correct_rate = float(final_match.group(3))

            checkpoint_epoch = eval_epoch - 1

            best_pq = None
            best_thr = None
            if current_pq:
                best_thr, best_pq = max(current_pq, key=lambda x: x[1])

            results.append({
                "epoch": checkpoint_epoch,
                "best_pq": best_pq,
                "threshold": best_thr,
                "avg_iou": avg_iou,
                "correct_rate": correct_rate,
            })

            current_pq = []


# sort high → low by PQ
results.sort(key=lambda x: x["best_pq"] if x["best_pq"] is not None else -1, reverse=True)
print("evaluating log:", LOG_PATH)

print("Ranked epochs by best PQ score")
print("-------------------------------------------------------")
print(f"{'rank':>4} {'epoch':>8} {'pq':>8} {'thr':>6} {'corr_rate':>10} {'avg_iou':>10}")
print("-------------------------------------------------------")

for i, r in enumerate(results, 1):
    pq = f"{r['best_pq']:.4f}" if r["best_pq"] else "N/A"
    thr = f"{r['threshold']:.2f}" if r["threshold"] else "N/A"

    print(
        f"{i:>4} {('epoch'+str(r['epoch'])):>8} {pq:>8} {thr:>6} "
        f"{r['correct_rate']:>10.4f} {r['avg_iou']:>10.4f}"
    )