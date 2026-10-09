mkdir -p ~/k450 && cp calib450.py ~/k450/ && cd ~/k450 && df -h ~ | tail -1
nohup setsid bash -c "/home/ubuntu/env/bin/python ~/k450/calib450.py ~/k450 8 > ~/k450/calib.log 2>&1" < /dev/null > /dev/null 2>&1 &
sleep 5; echo started
