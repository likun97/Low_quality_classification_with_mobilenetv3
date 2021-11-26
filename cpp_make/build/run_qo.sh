#!/bin/sh

ulimit -c unlimited

./plant_classify -edtv -f ../conf/client.cfg -k Classify -g ../log/lq_qo.log -n 1 -m 384

