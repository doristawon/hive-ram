# prints READY / FAIL / WAIT for tag $1 (run on the leader node)
cd $HOME/n1x-cluster; P=$(cat n1x_tun.pid)
if nsenter --preserve-credentials -U -n -t $P curl -s -m 3 localhost:18100/v1/models | grep -q object; then echo READY
elif grep -a -q 'Engine core initialization failed' m31-pp-$1-r0.log; then echo FAIL
else echo WAIT; fi
