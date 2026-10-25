# launch_rank.sh RANK TAG "EXTRA ENV" : start this node's vLLM rank inside the n1x_tun namespace (run as the normal user)
N1X=$HOME/n1x-cluster; cd $N1X
RANK=$1; TAG=$2; EXTRA=$3
if [ "$RANK" = 0 ]; then IP=10.99.0.2; else IP=10.99.0.1; fi
P=$(cat $N1X/n1x_tun.pid)
FADV_FOR_S=1200 setsid nohup python3 $(dirname "$0")/fadv.py $N1X/models/minimax-m31-sage215 3 > /dev/null 2>&1 < /dev/null &
INROOT_ENV="MODEL=/n1x/models/minimax-m31-sage215 VLLM_EXL3_MODEL_DIR=/n1x/models/minimax-m31-sage215 TFMOE=1 TFLIN=1 PYTHONPATH=/n1x/tfx:/n1x/m31prof TFX_BUILD=/n1x/tfx/build CPATH=/n1x/tfx/inc PP=2 NNODES=2 NODE_RANK=$RANK MASTER=10.99.0.2 HOST_IP=$IP PP_SPLIT=${PP_SPLIT:-42,18} GMU=0.80 UMA_ATS=0 $EXTRA" \
  setsid nohup nsenter --preserve-credentials -U -n -t $P ./inroot.sh vllm-m31-rootfs bash /n1x/m31-src/m31_serve.sh > m31-pp-$TAG-r$RANK.log 2>&1 < /dev/null &
sleep 2; echo "rank $RANK launched in tunnel ns $P"
