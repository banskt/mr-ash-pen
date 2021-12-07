module global_parameters
    use env_precision
    implicit none
    real(r8k), parameter :: d_zero = 0.d0
    real(r8k), parameter :: d_one = 1.d0, d_half = 0.5d0, d_two = 2.d0
    real(r8k), parameter :: pi = atan(d_one) * 4
    real(r8k), parameter :: log2pi = log(d_two * pi)
end module global_parameters

