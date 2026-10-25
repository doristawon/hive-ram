# tunrestart.sh server|client : restart this end of the n1x tunnel with the new n1x_tun.py
N1X=$HOME/n1x-cluster; cd $N1X
for p in $(ps -eo pid,args | awk '/n1x_tun.py/ && !/awk/ && !/tunrestart/ {print $1}'); do kill $p 2>/dev/null; done; sleep 1
cp /tmp/tunpkg/n1x_tun.py $N1X/n1x_tun.py; rm -f $N1X/n1x_tun.pid
if [ "$1" = server ]; then ARGS="server 5201 10.99.0.1/30"; else ARGS="client 10.77.0.1:5201 10.99.0.2/30"; fi
setsid nohup python3 $N1X/n1x_tun.py $ARGS > $N1X/n1x_tun.log 2>&1 < /dev/null &
sleep 3; tail -3 $N1X/n1x_tun.log; ls $N1X/n1x_tun.pid 2>/dev/null
