#!/bin/sh
 
dep_files="*_lq_qo_rethead_adpad.sh" 
# sshpass -p gpufirst@123 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r  $dep_files root@10.141.112.101:
# sshpass -p gpufirst@123 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r  $dep_files root@10.141.112.102: 
sshpass -p Gpu@2020 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r  $dep_files 10.141.112.50:
sshpass -p Gpu@2020 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r  $dep_files 10.141.112.51:
# sshpass -p Gpu@2020 scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -r  $dep_files 10.141.112.54:

