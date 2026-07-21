#!/bin/bash

echo "=== Hostname ==="
hostname

echo ""
echo "=== Kernel Version ==="
uname -r

echo ""
echo "=== CPU Info ==="
lscpu | grep -E 'Model name|Architecture|CPU\(s\)|MHz'

echo ""
echo "=== Memory Usage ==="
free -h
