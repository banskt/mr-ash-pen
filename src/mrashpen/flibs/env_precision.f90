module env_precision
    implicit none
    integer, parameter :: r4k  = selected_real_kind(6, 37) 
    integer, parameter :: r8k  = selected_real_kind(15, 307)
    integer, parameter :: r16k = selected_real_kind(33, 4931)
    integer, parameter :: i2k  = selected_int_kind(4)
    integer, parameter :: i4k  = selected_int_kind(9)
    integer, parameter :: i8k  = selected_int_kind(15)
end module env_precision
