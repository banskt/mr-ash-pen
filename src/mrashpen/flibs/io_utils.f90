subroutine print_array2d(x, m, n)
!
!   x is an input array of size (m, n)
!
    use env_precision
    implicit none
    integer, intent(in)   :: m, n
    real(r8k), dimension(m, n), intent(in)  :: x
    character(len=100)    :: fmt1
!
!   Format for each row should be:
!   fmt1 = '(n(2X,F7.2))'
!
    write (fmt1, '(A,I0,A)') '(', n, '(2X, F7.3))'
    write (6, fmt1) transpose(x)
!
end subroutine print_array2d

subroutine print_vector(x, m)
    use env_precision
    implicit none
    integer, intent(in)   :: m
    real(r8k), dimension(m), intent(in)  :: x
    character(len=100)    :: fmt1
!
!   Format for each row should be:
!   fmt1 = '(m(2X,F7.2))'
!
    write (fmt1, '(A,I0,A)') '(', m, '(2X, F7.3))'
    write (6, fmt1) x
!
end subroutine print_vector
