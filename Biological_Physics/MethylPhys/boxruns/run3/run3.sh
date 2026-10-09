set -uo pipefail
df -h $HOME | tail -1
cd ~/IAM-Validation && git pull -q origin main && git log --oneline -1 | cut -c1-50; cd - >/dev/null
/home/ubuntu/env/bin/python run3.py 2>&1 | grep -v -i warn | tail -12
rm -rf ~/run3/*/idat ~/run3/*/*.tar
