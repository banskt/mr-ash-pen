module futils
    use env_precision
    use global_parameters
    implicit none

contains

    subroutine fill_real_vector(x, a)
        real(r8k), intent(inout) :: x(:)
        real(r8k) :: a
        integer(i4k) :: i
        do i = 1, size(x)
            x(i) = a
        end do
    end subroutine fill_real_vector

    subroutine fill_real_matrix(x, a)
        real(r8k), intent(inout) :: x(:, :)
        real(r8k) :: a
        integer(i4k) :: i, j
        do j = 1, size(x, 2)
            do i = 1, size(x, 1)
                x(i, j) = a
            end do
        end do
    end subroutine fill_real_matrix

    function log_sum_exponent2d(z) result(zsum)
        real(r8k), intent(in)  :: z(:,:)
        real(r8k)              :: zsum(size(z, 1))
        real(r8k)              :: zmax, rowsum
        integer(i4k)           :: i
        real(r8k)              :: zT(size(z, 2), size(z, 1))
!        real(r8k)              :: tmp1(size(z, 2))
!       use transpose because Fortran use column-major ordering
        zT = transpose(z)
        call fill_real_vector(zsum, d_zero)
        do i = 1, size(z, 1)
            zmax    = maxval(zT(:, i))
            !write (6, *) "zmax:", zmax
            !tmp1    = zT(:, i) - zmax
            !write (6, *) i, "zi - zmax"
            !call print_vector(tmp1, size(tmp1))
            !tmp1    = vector_exp(tmp1)
            !write (6, *) "exponents"
            !call print_vector(tmp1, size(tmp1))
            rowsum  = sum(vector_exp(zT(:, i) - zmax))
            zsum(i) = log(rowsum) + zmax
        end do
    end function

    function vector_exp(x) result (y)
        real(r8k), intent(in) :: x(:)
        real(r8k)             :: y(size(x))
        integer(i4k)          :: i
        do i = 1, size(x)
            y(i) = exp(x(i))
        end do
    end function


    function vector_log(x) result (y)
        real(r8k), intent(in) :: x(:)
        real(r8k)             :: y(size(x))
        integer(i4k)          :: i
        do i = 1, size(x)
            y(i) = log(x(i))
        end do
    end function

    function duplicate_columns(x, ncol) result(y)
        integer(i4k), intent(in) :: ncol
        real(r8k), intent(in)    :: x(:)
        real(r8k)                :: y(size(x), ncol)
        integer(i4k)             :: i
        do i = 1, ncol
            y(:, i) = x
        end do
    end function duplicate_columns

    function get_nonzero_index_vector(x) result(vidx)
        real(r8k), intent(in)  :: x(:)
        integer                :: vidx(size(x))
        integer                :: i
        do i = 1, size(x)
            vidx(i) = 0
            if (x(i) .ne. d_zero) then
                vidx(i) = 1
            end if
        end do
    end function get_nonzero_index_vector

    subroutine print_array2d(x, m, n)
!
!       x is an input array of size (m, n)
!
        implicit none
        integer, intent(in)   :: m, n
        real(r8k), dimension(m, n), intent(in)  :: x
        character(len=100)    :: fmt1
!
!       Format for each row should be:
!       fmt1 = '(n(2X,F7.2))'
!
        write (fmt1, '(A,I0,A)') '(', n, '(2X, F7.3))'
        write (6, fmt1) transpose(x)
!
    end subroutine print_array2d

    subroutine print_vector(x, m)
        implicit none
        integer, intent(in)   :: m
        real(r8k), dimension(m), intent(in)  :: x
        character(len=100)    :: fmt1
!
!       Format for each row should be:
!       fmt1 = '(m(2X,F7.2))'
!
        write (fmt1, '(A,I0,A)') '(', m, '(2X, F20.3))'
        write (6, fmt1) x
!
    end subroutine print_vector

end module futils
