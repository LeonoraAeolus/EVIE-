#!/usr/bin/env bash
# Start the GPU VM again and print its IP (a dynamic public IP can change after >1 h stopped).
set -euo pipefail
VM_NAME="${VM_NAME:-openmath}"
VM_ID=$(nebius compute instance get-by-name --name "$VM_NAME" --format jsonpath='{.metadata.id}')
nebius compute instance start --id "$VM_ID"
IP=$(nebius compute instance get-by-name --name "$VM_NAME" --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]')
echo "Started $VM_NAME at $IP. Tunnel: ssh -N -L 8000:localhost:8000 ${SSH_USER:-tesla}@$IP"
echo "vLLM needs ~1-3 min to load the model after boot (weights are already on disk)."
