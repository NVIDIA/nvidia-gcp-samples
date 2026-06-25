#!/usr/bin/env bash

set -euo pipefail

# Set up ThinLinc Server for an accelerated Isaac Sim desktop by installing
# VirtualGL and the ThinLinc server packages.

sudo apt-get update
sudo apt-get remove -y needrestart || true
sudo apt-get install -y curl unzip wget openssl

wget -q https://github.com/VirtualGL/virtualgl/releases/download/3.1.4/virtualgl_3.1.4_amd64.deb
sudo apt-get install -y ./virtualgl_3.1.4_amd64.deb

wget -q https://www.cendio.com/downloads/server/tl-4.20.1-server.zip
unzip -o tl-4.20.1-server.zip
sudo apt-get install -y ./tl-4.20.1-server/packages/thinlinc-server_4.20.1-4529_amd64.deb

EXTERNAL_IP=$(curl -s ifconfig.me)
OUTPUT_FILE="/tmp/thinlinc-answer-file"

UNUSED_PASS=$(openssl rand -base64 12)
ADMIN_HASH=$(/opt/thinlinc/sbin/tl-gen-auth "${UNUSED_PASS}")

cat > "${OUTPUT_FILE}" << EOF
# ThinLinc tl-setup answers file
accept-eula=yes
server-type=master
migrate-conf=ignore
install-required-libs=yes
install-nfs=no
install-sshd=no
install-gtk=yes
install-python-ldap=no
agent-hostname-choice=ip
manual-agent-hostname=${EXTERNAL_IP}
email-address=sample@example.com
tlwebadm-password=${ADMIN_HASH}
setup-thinlocal=no
setup-nearest=no
setup-firewall-ssh=no
setup-firewall-tlwebaccess=no
setup-firewall-tlwebadm=no
setup-firewall-tlmaster=no
setup-firewall-tlagent=no
setup-selinux=no
setup-apparmor=no
missing-answer=abort
EOF

sudo chmod 600 "${OUTPUT_FILE}"
sudo /opt/thinlinc/sbin/tl-setup -a "${OUTPUT_FILE}"

CONF_FILE="/opt/thinlinc/etc/conf.d/vsmagent.hconf"
sudo sed -i \
  -e "s/^master_hostname=.*/master_hostname=${EXTERNAL_IP}/" \
  -e "s/^agent_hostname=.*/agent_hostname=${EXTERNAL_IP}/" \
  "${CONF_FILE}"

sudo systemctl restart vsmagent

sudo rm -f /opt/thinlinc/etc/xstartup.d/40-tl-mount-localdrives
sudo rm -f /opt/thinlinc/etc/xlogout.d/tl-umount-localdrives
sudo rm -f /opt/thinlinc/etc/xstartup.d/50-tl-wait-smartcard

if id ubuntu >/dev/null 2>&1; then
  echo "gsettings set org.gnome.mutter check-alive-timeout 0" | sudo tee -a /home/ubuntu/.profile >/dev/null
fi

echo "ThinLinc setup complete. Open https://${EXTERNAL_IP}:300"
