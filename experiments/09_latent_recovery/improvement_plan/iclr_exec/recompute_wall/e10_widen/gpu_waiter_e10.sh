#!/bin/bash
# E10 WIDEN auto-launch waiter: polls (120s) for a FULLY-FREE GPU (0 MiB used),
# confirms it stays free across a second 120s check, then launches run_e10_queue.sh
# pinned to it. Launches at most once (launch.lock). Runs on skampere2 under nohup;
# independent of the laptop's Kerberos. Only ever claims a 0-MiB device -- never
# touches a GPU a lab-mate is using.
set -u
E10DIR=/lfs/skampere2/0/eobbad/free-energy/experiments/09_latent_recovery/improvement_plan/iclr_exec/recompute_wall/e10_widen
LOCK=$E10DIR/launch.lock
LOG=$E10DIR/logs/waiter.log
POLL=120
mkdir -p "$E10DIR/logs"
if [ -e "$LOCK" ]; then echo "$(date -Is) lock exists, exiting" >> "$LOG"; exit 0; fi
echo "$(date -Is) waiter started (pid $$), poll=${POLL}s" >> "$LOG"
while :; do
  free=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2==0{print $1; exit}')
  if [ -n "$free" ]; then
    echo "$(date -Is) GPU $free reads 0 MiB; confirming stable for ${POLL}s" >> "$LOG"
    sleep "$POLL"
    still=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$free" | tr -d ' ')
    if [ "$still" = "0" ]; then
      if ( set -o noclobber; : > "$LOCK" ) 2>/dev/null; then   # atomic launch-once
        echo "$(date -Is) GPU $free free+stable, launching queue" >> "$LOG"
        cd "$E10DIR" || exit 1
        CUDA_VISIBLE_DEVICES=$free nohup bash run_e10_queue.sh > queue.out 2>&1 &
        echo $! > "$E10DIR/queue.pid"
        echo "$(date -Is) queue launched, pid $(cat "$E10DIR/queue.pid") on GPU $free" >> "$LOG"
      else
        echo "$(date -Is) lock appeared during confirm; another launcher won, exiting" >> "$LOG"
      fi
      exit 0
    fi
    echo "$(date -Is) GPU $free was claimed within ${POLL}s window" >> "$LOG"
  fi
  sleep "$POLL"
done
