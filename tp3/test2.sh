for i in 1 2 4 8 16 32 64 128 256 512 1024 ; do
  for j in 1 2 4 8 16 32 64 128 256 512 1024 ; do
    if [[ $(( i * j )) -le 1024 ]]; then
      make BLOCK_SIZE_X=$i BLOCK_SIZE_Y=$j 1>&2
      ./MatrixProduct -c GPU -gpu-k 2 | tail -n 4 | head -n 2 | tr -s ' ' | rev | cut -d' ' -f 1 | rev | xargs echo $i $j | tr ' ' ,
    fi
  done
done
