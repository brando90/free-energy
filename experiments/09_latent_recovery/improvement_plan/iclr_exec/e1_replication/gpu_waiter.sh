#!/bin/bash
# E1 auto-launch waiter: polls for a fully-free GPU (0 MiB used), confirms it stays
# free for 60s, then launches run_e1_queue.sh pinned to it. Launches at most once
# (launch.lock). Runs on skampere2 under nohup; independent of the laptop's Kerberos.
E1DIR=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/e1_replication
LOCK=$E1DIR/launch.lock
LOG=$E1DIR/logs/waiter.log
mkdir -p "$E1DIR/logs"
if [ -e "$LOCK" ]; then echo "$(date -Is) lock exists, exiting" >> "$LOG"; exit 0; fi
echo "$(date -Is) waiter started (pid $$)" >> "$LOG"
while :; do
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1; exit}')
  if [ -n "$free" ]; then
    sleep 60
    still=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$free" | tr -d ' ')
    if [ "$still" = "0" ]; then
      touch "$LOCK"
      echo "$(date -Is) GPU $free free and stable, launching queue" >> "$LOG"
      cd "$E1DIR" || exit 1
      CUDA_VISIBLE_DEVICES=$free nohup bash run_e1_queue.sh > queue.out 2>&1 &
      echo $! > "$E1DIR/queue.pid"
      echo "$(date -Is) queue launched, pid $(cat "$E1DIR/queue.pid")" >> "$LOG"
      exit 0
    fi
    echo "$(date -Is) GPU $free was free but got claimed within 60s" >> "$LOG"
  fi
  sleep 300
done
