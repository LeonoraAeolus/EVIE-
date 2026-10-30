#!/usr/bin/env bash
# Stop the GPU VM properly (from your laptop). Stopped = no GPU/CPU/RAM charges; the disk is still billed.
# Do NOT use `sudo shutdown` on the VM: Nebius treats it as a failure, restarts the VM and keeps billing
# (https://docs.nebius.com/compute/virtual-machines/stopped-with-linux-commands).
set -euo pipefail
VM_NAME="${VM_NAME:-openmath}"
VM_ID=$(nebius compute instance get-by-name --name "$VM_NAME" --format jsonpath='{.metadata.id}')
nebius compute instance stop --id "$VM_ID"
echo "Stopped $VM_NAME ($VM_ID). Start again with ./start_vm.sh"
