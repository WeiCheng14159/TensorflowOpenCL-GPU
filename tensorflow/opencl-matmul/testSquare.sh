#!/system/bin/sh
# Time testing bash running on Android
END=50
START=4
for i in $(seq $START $END)
do

  # TransposeA: No, TransposeB: No,
  echo "MatMul A=[$i,$i] B=[$i,$i] "
  ./opencl-matmul $i $i $i $i 0 0 1

  # TransposeA: Yes, TransposeB: No,
  # echo "MatMul A=[$i,$i] B=[$i,$i] "
  # ./opencl-matmul $i $i $i $i 1 0 1

  # TransposeA: No, TransposeB: Yes,
  # echo "MatMul A=[$i,$i] B=[$i,$i] "
  # ./opencl-matmul $i $i $i $i 0 1 1

  # TransposeA: Yes, TransposeB: Yes,
  # echo "MatMul A=[$i,$i] B=[$i,$i] "
  # ./opencl-matmul $i $i $i $i 1 1 1

done
