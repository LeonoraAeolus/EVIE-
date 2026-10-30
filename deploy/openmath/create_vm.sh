#!/usr/bin/env bash
# Create a Nebius AI Cloud GPU VM that serves OpenMath-Nemotron with vLLM (OpenAI-compatible API).
# Run this on YOUR laptop, after installing + configuring the Nebius CLI:
#   curl -sSL https://artifacts.nebius.cloud/cli/install.sh | bash      (https://docs.nebius.com/cli/install)
#   nebius profile create                                               (log in, pick your project)
# Usage:
#   ./create_vm.sh                      # 7B on 1x L40S 48 GB (recommended)
#   MODEL_SIZE=14B PRESET=1gpu-16vcpu-64gb ./create_vm.sh
# Commands follow https://docs.nebius.com/compute/quickstart-host-model (checked 2026-10-13).
# NOT run by Tab: it needs your Nebius account. Read the README first. Billing starts when the VM starts.
set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-7B}"                    # 1.5B | 7B | 14B | 32B
MODEL="nvidia/OpenMath-Nemotron-${MODEL_SIZE}"
VM_NAME="${VM_NAME:-openmath}"
PLATFORM="${PLATFORM:-gpu-l40s-a}"                # L40S 48 GB + Intel (eu-north1). gpu-h100-sxm for an H100 80 GB
PRESET="${PRESET:-1gpu-8vcpu-32gb}"               # H100: 1gpu-16vcpu-200gb
DISK_GB="${DISK_GB:-100}"                         # weights (15 GB for 7B, 30 GB for 14B) + vLLM + cache
IMAGE_FAMILY="${IMAGE_FAMILY:-ubuntu24.04-cuda12}" # Ubuntu with NVIDIA drivers preinstalled
SSH_USER="${SSH_USER:-tesla}"                     # not root/admin (reserved on Nebius)
SSH_PUBKEY="${SSH_PUBKEY:-$HOME/.ssh/id_ed25519.pub}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"                   # 1 = spot VM: about half price, can be stopped by Nebius any time
API_KEY="${OPENMATH_API_KEY:-$(openssl rand -hex 24)}"

command -v nebius >/dev/null || { echo "Install the Nebius CLI first (see header)"; exit 1; }
command -v jq >/dev/null || { echo "Install jq first"; exit 1; }
[ -f "$SSH_PUBKEY" ] || { echo "No SSH key at $SSH_PUBKEY (ssh-keygen -t ed25519)"; exit 1; }

HERE="$(cd "$(dirname "$0")" && pwd)"
SERVE_B64="$(base64 < "$HERE/serve.sh" | tr -d '\n')"

echo "Creating ${DISK_GB} GiB boot disk (${IMAGE_FAMILY})..."
BOOT_DISK_ID=$(nebius compute disk create \
  --name "${VM_NAME}-boot" \
  --size-gibibytes "$DISK_GB" \
  --type network_ssd \
  --source-image-family-image-family "$IMAGE_FAMILY" \
  --block-size-bytes 4096 \
  --format jsonpath='{.metadata.id}')

SUBNET_ID=$(nebius vpc subnet list --format jsonpath='{.items[0].metadata.id}')

USER_DATA=$(jq -Rrs '.' <<EOF
#cloud-config
users:
  - name: ${SSH_USER}
    sudo: ALL=(ALL) NOPASSWD:ALL
    shell: /bin/bash
    ssh_authorized_keys:
      - $(cat "$SSH_PUBKEY")
write_files:
  - path: /opt/openmath/serve.sh
    permissions: "0755"
    encoding: b64
    content: ${SERVE_B64}
  - path: /etc/openmath.env
    permissions: "0600"
    content: |
      OPENMATH_MODEL=${MODEL}
      OPENMATH_API_KEY=${API_KEY}
      MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
      BIND_HOST=${BIND_HOST:-127.0.0.1}
  - path: /etc/systemd/system/openmath.service
    content: |
      [Unit]
      Description=OpenMath-Nemotron (vLLM, OpenAI-compatible)
      After=network-online.target
      [Service]
      EnvironmentFile=/etc/openmath.env
      ExecStart=/opt/openmath/serve.sh
      Restart=on-failure
      User=${SSH_USER}
      [Install]
      WantedBy=multi-user.target
runcmd:
  - [ bash, -c, "mkdir -p /opt/openmath && chown -R ${SSH_USER} /opt/openmath && systemctl daemon-reload && systemctl enable --now openmath.service" ]
EOF
)

SPOT_ARGS=()
if [ "$PREEMPTIBLE" = "1" ]; then
  # https://docs.nebius.com/compute/virtual-machines/preemptible
  SPOT_ARGS=(--follows-spot-price --recovery-policy fail --preemptible-on-preemption stop)
fi

echo "Creating VM ${VM_NAME} on ${PLATFORM} / ${PRESET} (billing starts now)..."
VM_ID=$(nebius compute instance create \
  --name "$VM_NAME" \
  --resources-platform "$PLATFORM" \
  --resources-preset "$PRESET" \
  --boot-disk-existing-disk-id "$BOOT_DISK_ID" \
  --boot-disk-attach-mode READ_WRITE \
  --cloud-init-user-data "$USER_DATA" \
  --network-interfaces "[{\"name\": \"default-subnet\", \"subnet_id\": \"$SUBNET_ID\", \"ip_address\": {}, \"public_ip_address\": {}}]" \
  ${SPOT_ARGS[@]+"${SPOT_ARGS[@]}"} \
  --format jsonpath='{.metadata.id}')

IP=$(nebius compute instance get-by-name --name "$VM_NAME" --format json \
  | jq -r '.status.network_interfaces[0].public_ip_address.address | split("/")[0]')

cat <<MSG

VM id:   $VM_ID
IP:      $IP
Model:   $MODEL  (first start downloads the weights and installs vLLM: allow ~10-15 min)

1) Open a tunnel (keeps the API off the public internet):
     ssh -N -L 8000:localhost:8000 ${SSH_USER}@${IP}
2) In tesla-lab/.env:
     OPENMATH_BASE_URL=http://localhost:8000/v1
     OPENMATH_MODEL=${MODEL}
     OPENMATH_API_KEY=${API_KEY}
3) Check it:  python scripts/check_openmath.py
4) Watch the server log:  ssh ${SSH_USER}@${IP} journalctl -u openmath -f

STOP THE VM WHEN YOU ARE DONE (Linux 'shutdown' does NOT stop billing on Nebius):
     ./stop_vm.sh        or   nebius compute instance stop --id $VM_ID
MSG
