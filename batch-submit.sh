#!/bin/bash
#SBATCH --job-name=throttled_submit
#SBATCH -p kibr
#SBATCH --output=logs/throttle_%j.out
#SBATCH --error=logs/throttle_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=1000
#SBATCH -c 1

# ------------------------ Config ------------------------
START=0
END=28206
CHUNK_SIZE=1000
MAX_ACTIVE_JOBS=2900
STATE_FILE="submit_chunks_state.txt"
JOB_SCRIPT="./main.sh"      # Updated job script
RESUBMIT_DELAY="now + 5 minutes"    # Valid SLURM format

# ------------------------ Init ------------------------
if [ ! -f "$STATE_FILE" ]; then
    echo "$START" > "$STATE_FILE"
fi

CURRENT=$(cat "$STATE_FILE")

# ------------------------ Completion Check ------------------------
if [ "$CURRENT" -gt "$END" ]; then
    echo "All chunks submitted. Cleaning up."
    rm -f "$STATE_FILE"
    exit 0
fi

# ------------------------ Job Throttling ------------------------
ACTIVE_JOBS=$(squeue -u "$USER" | grep -c -v "JOBID")
echo "$(date): Active jobs = $ACTIVE_JOBS"

if [ "$ACTIVE_JOBS" -lt "$MAX_ACTIVE_JOBS" ]; then
    CHUNK_START=$CURRENT
    CHUNK_END=$((CURRENT + CHUNK_SIZE - 1))
    if [ "$CHUNK_END" -gt "$END" ]; then
        CHUNK_END=$END
    fi

    echo "Submitting chunk: $CHUNK_START-$CHUNK_END"
    sbatch --array=${CHUNK_START}-${CHUNK_END} "$JOB_SCRIPT"

    NEXT=$((CHUNK_END + 1))
    echo "$NEXT" > "$STATE_FILE"
else
    echo "Too many active jobs. Skipping submission this round."
fi

# ------------------------ Reschedule Self ------------------------
echo "Scheduling next check in $RESUBMIT_DELAY..."
sbatch --begin="$RESUBMIT_DELAY" "$0"
