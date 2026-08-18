#!/bin/bash
set -e

# Install UltraLeap repository and key
wget -qO - https://repo.ultraleap.com/keys/apt/gpg | gpg --dearmor | sudo tee /etc/apt/trusted.gpg.d/ultraleap.gpg

echo 'deb [arch=amd64] https://repo.ultraleap.com/apt stable main' | sudo tee /etc/apt/sources.list.d/ultraleap.list
sudo apt update

# UltraLeap software is governed by its vendor EULA. Review and accept the
# terms yourself when the package manager prompts; this script does not accept
# them on your behalf. If no prompt appears, run `leapctl eula` before use.
sudo apt install ultraleap-hand-tracking

# Clone and install leapc-python-bindings
git clone https://github.com/ultraleap/leapc-python-bindings /tmp/leapc-python-bindings
cd /tmp/leapc-python-bindings
pip install -r requirements.txt
python -m build leapc-cffi
pip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz
pip install -e leapc-python-api
