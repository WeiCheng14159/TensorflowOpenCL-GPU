// This is a simple OpenCL kernel

void kernel multiply_by(global int* A, const int c) {
    A[get_global_id(0)] = c * A[get_global_id(0)];
}
