for i in 1 2 4 8 16 32 64 128 256 512 1024 ; do
  #echo "Running test with i="$i
  make BLOCK_SIZE_X=$i BLOCK_SIZE_Y=$i 1>&2
  ./MatrixProduct -c GPU -gpu-k 1 | tail -n 4 | head -n 2 | tr -s ' ' | rev | cut -d' ' -f 1 | rev | xargs echo $i | tr ' ' ,
done
