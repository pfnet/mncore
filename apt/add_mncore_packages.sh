#! /bin/bash
set -eu -o pipefail

if [[ ${UID} != 0 ]]; then
    echo "E: this script needs root" >&2
	exit 1
fi

. /etc/os-release

DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends ca-certificates wget
apt-get clean

wget https://asia-northeast1-apt.pkg.dev/doc/repo-signing-key.gpg -O /usr/share/keyrings/mncore-packages-archive-keyring.gpg.asc
chmod 644 /usr/share/keyrings/mncore-packages-archive-keyring.gpg.asc

echo "deb [signed-by=/usr/share/keyrings/mncore-packages-archive-keyring.gpg.asc] https://asia-northeast1-apt.pkg.dev/projects/mncore-packages ${VERSION_CODENAME} main" | tee -a /etc/apt/sources.list.d/mncore-packages.list
apt-get update -y
