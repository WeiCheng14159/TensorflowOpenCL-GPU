#!/system/bin/sh
END=50
START=4
numTimes=1
for i in $(seq $START $END)
do
  for j in $(seq $START $END)
  do
    for k in $(seq $START $END)
    do

      # TransposeA: No, TransposeB: No,
      echo "MatMul A=[$i,$j] B=[$j,$k] "
      ./opencl-matmul $i $j $j $k 0 0 $numTimes

      # TransposeA: Yes, TransposeB: No,
      # echo "MatMul A=[$j,$i] B=[$j,$k] "
      # ./opencl-matmul $j $i $j $k 1 0 $numTimes

      # TransposeA: No, TransposeB: Yes,
      # echo "MatMul A=[$i,$j] B=[$j,$k] "
      # ./opencl-matmul $i $j $k $j 0 1 $numTimes

      # TransposeA: Yes, TransposeB: Yes,
      # echo "MatMul A=[$i,$j] B=[$j,$k] "
      # ./opencl-matmul $j $i $k $j 1 1 $numTimes

    done
  done
done
