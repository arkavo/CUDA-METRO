#!/bin/bash
# Am I allowed to run (GPU) jobs on this cluster?
# Read-only except for the final optional 1-minute test job.
# Run on the LOGIN node:  bash check_slurm.sh

echo "=============================================="
echo " 1. Is Slurm here at all?"
echo "=============================================="
if ! command -v sinfo >/dev/null 2>&1; then
    echo "  no sinfo on PATH."
    echo "  -> either Slurm isn't the scheduler (check for qsub=PBS/Torque,"
    echo "     bsub=LSF, qstat) or you need: module load slurm"
    command -v qsub  >/dev/null && echo "  found qsub  -> this is PBS/Torque, not Slurm"
    command -v bsub  >/dev/null && echo "  found bsub  -> this is LSF, not Slurm"
    exit 1
fi
sinfo --version
echo

echo "=============================================="
echo " 2. Which partitions exist, and can I use them?"
echo "=============================================="
sinfo -o "%20P %10a %10l %6D %10T %N" | head -25
echo
echo "  (AVAIL=up means the partition is running; a partition you may not"
echo "   submit to will still be listed here - restrictions show in step 4.)"
echo

echo "=============================================="
echo " 3. Which partitions actually have GPUs?"
echo "=============================================="
sinfo -o "%20P %10G %N" | grep -iv "(null)" | head -15
echo "  (empty/no gres column => no GPUs advertised on that partition)"
echo

echo "=============================================="
echo " 4. Do I have an account/association? (the usual blocker)"
echo "=============================================="
if command -v sacctmgr >/dev/null 2>&1; then
    sacctmgr -n show assoc user="$USER" format=Account%20,Partition%20,QOS%30,MaxJobs 2>&1 | head -20
    echo
    echo "  If this is EMPTY you have no association -> submission will be"
    echo "  rejected with 'Invalid account or account/partition combination'."
    echo "  Ask the helpdesk to add you to a project/allocation."
    echo
    echo "  If an Account is listed, you may need to pass it explicitly:"
    echo "      sbatch --account=<that account> ..."
else
    echo "  sacctmgr not available (accounting may be off - often means no"
    echo "  account flag is needed)."
fi
echo

echo "=============================================="
echo " 5. Fairshare / allocation left"
echo "=============================================="
command -v sshare >/dev/null 2>&1 && sshare -U 2>&1 | head -8 || echo "  sshare not available"
echo

echo "=============================================="
echo " 6. Restrictions on the GPU partitions"
echo "=============================================="
for p in $(sinfo -h -o "%P" | tr -d '*' | sort -u); do
    info=$(scontrol show partition "$p" 2>/dev/null)
    gres=$(echo "$info" | grep -o "TRES=[^ ]*gpu[^ ]*")
    [ -z "$gres" ] && continue
    echo "--- $p"
    echo "$info" | grep -oE "AllowGroups=[^ ]+|AllowAccounts=[^ ]+|AllowQos=[^ ]+|MaxTime=[^ ]+|DefaultTime=[^ ]+" \
        | sed 's/^/    /'
done
echo "  (AllowGroups/AllowAccounts=ALL means open to you; a named group means"
echo "   you must be a member - check with: groups)"
echo
echo "  your groups: $(groups)"
echo

echo "=============================================="
echo " 7. Anything queued/running under your name"
echo "=============================================="
squeue -u "$USER" 2>&1 | head -5
echo

echo "=============================================="
echo " THE DEFINITIVE TEST"
echo "=============================================="
echo "Reasoning about the above is guesswork; submitting is proof."
echo "Edit the partition name, then run:"
echo
echo "  sbatch --partition=<gpu-partition> --gres=gpu:1 --time=00:02:00 \\"
echo "         --wrap='hostname; nvidia-smi; echo SLURM-GPU-OK'"
echo
echo "then:  squeue -u $USER      # watch it"
echo "       cat slurm-<jobid>.out"
echo
echo "If it prints a GPU table and SLURM-GPU-OK, you are fully set up."
echo "Common rejections and what they mean:"
echo "  'Invalid account...'          -> add --account=<acct> (step 4)"
echo "  'Invalid partition name'      -> wrong partition (step 2)"
echo "  'Requested node config not available' -> that partition has no such GPU;"
echo "                                   try --gres=gpu:1 without a type, or"
echo "                                   name the type e.g. --gres=gpu:a100:1"
echo "  'User's group not permitted'  -> not in AllowGroups (step 6)"
echo "  stays PENDING with (QOSMaxJobs/Priority) -> allowed, just queued"
