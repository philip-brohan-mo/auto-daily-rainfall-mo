#!/usr/bin/env bash
# Checkpoint Management Guide
#
# This document explains how to:
# 1. Fine-tune a model on Azure
# 2. Register the checkpoint
# 3. Extract using the fine-tuned checkpoint
# 4. Compare results across different checkpoints

# ── Fine-tuning a Model ───────────────────────────────────────────────────────
# Syntax: bash scripts/aml_submit.sh [--dataset real|fake] [--model MODEL] finetune
#
# Examples:
#   # Fine-tune Granite 3.2 on fake data
#   bash scripts/aml_submit.sh --dataset fake --model granite finetune
#
#   # Fine-tune Granite 4.1 on combined data
#   bash scripts/aml_submit.sh --dataset real --model granite4 finetune
#
#   # Fine-tune SmolVLM2 with custom dataset paths
#   bash scripts/aml_submit.sh \
#       --images-path my_data/images \
#       --transcriptions-path my_data/transcriptions \
#       --model smolvlm2 \
#       finetune
#
# OUTPUT:
#   - Azure job is submitted
#   - Job ID is printed
#   - Instructions are shown for registering the checkpoint after it completes

# ── Registering a Checkpoint ──────────────────────────────────────────────────
# After the Azure job completes, the checkpoint is saved to:
#   outputs/checkpoints/{model}-{dataset}-{timestamp}/
#
# Register it with:
#   python scripts/create_model_registry_entry.py \
#       --checkpoint-path outputs/checkpoints/granite-fake-20260526-143000 \
#       --base-model granite \
#       --dataset fake \
#       --registry-file outputs/model_registry.json \
#       --notes "Training on 998 synthetic images"
#
# The registry file (outputs/model_registry.json) tracks all trained models.

# ── Listing Registered Checkpoints ────────────────────────────────────────────
# bash scripts/list_checkpoints.sh                    # Human-readable table
# bash scripts/list_checkpoints.sh --format json      # JSON format
#
# OUTPUT example:
#   Registered Checkpoints:
#   ───────────────────────────────────────────────────────────────────────────────────────────
#   Checkpoint Name                          Base Model      Dataset    Created
#   ─────────────────────────────────────────────────────────────────────────────────────────
#   granite-fake-20260526-143000             granite         fake       2026-05-26
#   granite4-fake-20260526-150000            granite4        fake       2026-05-26
#
#   Extract using a checkpoint:
#     1. bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/granite-fake-20260526-143000 extract
#     2. bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/granite4-fake-20260526-150000 extract

# ── Extracting with a Checkpoint ──────────────────────────────────────────────
# Syntax: bash scripts/aml_submit.sh [--dataset real|fake] --checkpoint PATH extract
#
# Examples:
#   # Extract from real data using a fine-tuned checkpoint
#   bash scripts/aml_submit.sh \
#       --dataset real \
#       --checkpoint outputs/checkpoints/granite-fake-20260526-143000 \
#       extract
#
#   # Quick test: extract 10 images from fake data with a checkpoint
#   bash scripts/aml_submit.sh \
#       --dataset fake \
#       --checkpoint outputs/checkpoints/granite-fake-20260526-143000 \
#       --limit 10 \
#       extract
#
#   # Extract full dataset using checkpoint, 8 shards on A100
#   bash scripts/aml_submit.sh \
#       --dataset real \
#       --checkpoint outputs/checkpoints/granite-fake-20260526-143000 \
#       --total-shards 8 \
#       --env-variant a100 \
#       extract

# ── Comparing Multiple Checkpoints ────────────────────────────────────────────
# To compare extraction quality across different checkpoints:
#
#   1. Register multiple fine-tuned checkpoints:
#      python scripts/create_model_registry_entry.py \
#          --checkpoint-path outputs/checkpoints/granite-fake-20260526-143000 \
#          --base-model granite --dataset fake --registry-file outputs/model_registry.json
#      
#      python scripts/create_model_registry_entry.py \
#          --checkpoint-path outputs/checkpoints/granite4-real-20260526-150000 \
#          --base-model granite4 --dataset real --registry-file outputs/model_registry.json
#
#   2. Extract using each checkpoint:
#      bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/granite-fake-20260526-143000 --limit 50 extract
#      bash scripts/aml_submit.sh --checkpoint outputs/checkpoints/granite4-real-20260526-150000 --limit 50 extract
#
#   3. Evaluate both extractions against ground truth:
#      bash scripts/aml_submit.sh --limit 50 evaluate

# ── Key Flags & Options ───────────────────────────────────────────────────────
# --dataset real|fake              : Quick dataset selector (default: real)
# --images-path PATH               : Custom images directory path
# --transcriptions-path PATH       : Custom transcriptions directory path
# --model MODEL                    : Model preset (smolvlm, smolvlm2, granite, granite4)
# --checkpoint PATH                : Use a fine-tuned checkpoint for extraction
# --limit N                        : Process only N images per shard (smoke tests)
# --total-shards N                 : Parallel shards (default: 8 for extract, 1 for finetune)
# --env-variant v100|a100          : GPU type (default: v100)
# --compute CLUSTER                : Azure ML compute cluster name
#
# Examples combining multiple flags:
#   # Fine-tune Granite4 on fake data
#   bash scripts/aml_submit.sh --dataset fake --model granite4 finetune
#
#   # Extract using the checkpoint on real data, with 50-image test
#   bash scripts/aml_submit.sh \
#       --dataset real \
#       --checkpoint outputs/checkpoints/granite4-fake-20260526-150000 \
#       --limit 50 \
#       extract
#
#   # Full extraction on A100 cluster with checkpoint
#   bash scripts/aml_submit.sh \
#       --dataset real \
#       --checkpoint outputs/checkpoints/granite4-fake-20260526-150000 \
#       --total-shards 8 \
#       --env-variant a100 \
#       --compute gpu-cluster-a100 \
#       extract

# ── Workflow Example: Fine-tune → Evaluate ────────────────────────────────────
# Goal: Fine-tune Granite 4.1 on fake data, then extract from real data and
# evaluate accuracy.
#
# Step 1: Submit fine-tuning job
#   bash scripts/aml_submit.sh --dataset fake --model granite4 finetune
#   # Job ID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
#   # Note the job ID and checkpoint location from output
#
# Step 2: Wait for job to complete (monitor in Azure ML Studio)
#
# Step 3: Register the checkpoint (after job completes)
#   python scripts/create_model_registry_entry.py \
#       --checkpoint-path outputs/checkpoints/granite4-fake-20260526-150000 \
#       --base-model granite4 \
#       --dataset fake \
#       --registry-file outputs/model_registry.json \
#       --notes "Fine-tuned on 998 synthetic images"
#
# Step 4: Extract from real data using the checkpoint (50-image test)
#   bash scripts/aml_submit.sh \
#       --dataset real \
#       --checkpoint outputs/checkpoints/granite4-fake-20260526-150000 \
#       --limit 50 \
#       extract
#   # Results in: outputs/extractions/granite4-fake-20260526-150000/YYYYMMDD-HHMMSS/
#
# Step 5: Evaluate extraction accuracy
#   bash scripts/aml_submit.sh --dataset real --limit 50 evaluate
#   # Compare results with the original model
#
# Step 6: List all checkpoints to compare multiple fine-tuned models
#   bash scripts/list_checkpoints.sh
#
# Step 7: Extract with other checkpoints and compare results

# ── Troubleshooting ───────────────────────────────────────────────────────────
#
# Q: I got "Unknown option: --checkpoint" error
# A: Make sure you're using the latest version of aml_submit.sh (May 2026+)
#    Run: git pull origin main  (if in git repo)
#
# Q: Checkpoint path not found when extracting
# A: Make sure the path is relative to the Azure datastore root
#    Example: outputs/checkpoints/granite-fake-20260526-143000 (not full path)
#    The full URI is constructed as: azureml://datastores/.../outputs/checkpoints/...
#
# Q: Where is the fine-tuned checkpoint saved?
# A: In outputs/checkpoints/ on the Azure datastore
#    You can browse it in Azure ML Studio → Data → Datastore browse
#
# Q: How do I use a locally downloaded checkpoint?
# A: Copy it to the Azure datastore or use --images-path to point to it locally
#    For local testing: bash scripts/aml_submit.sh --model /path/to/checkpoint --limit 5 extract
#
# Q: Can I fine-tune on multiple datasets together?
# A: Create a custom dataset directory and use --images-path and --transcriptions-path flags:
#    bash scripts/aml_submit.sh \
#        --images-path combined_data/images \
#        --transcriptions-path combined_data/transcriptions \
#        --model granite4 \
#        finetune

echo "See comments in this file for checkpoint management workflows."
