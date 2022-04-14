module futils
    use env_precision
    use global_parameters
    implicit none

contains

!    subroutine log_sum_exponent2d(z, zsum)
!        real(r8k), intent(in)  :: z(:,:)
!        real(r8k), intent(out) :: zsum(size(z, 1))
!        real(r8k)              :: zmax, rowsum
!        integer(i4k)           :: i
!        real(r8k)              :: zT(size(z, 2), size(z, 1))
!!        real(r8k)              :: tmp1(size(z, 2))
!!       use transpose because Fortran use column-major ordering
!        zT = transpose(z)
!        zsum = d_zero
!        do i = 1, size(z, 1)
!            zmax    = maxval(zT(:, i))
!            !write (6, *) "zmax:", zmax
!            !tmp1    = zT(:, i) - zmax
!            !write (6, *) i, "zi - zmax"
!            !call print_vector(tmp1, size(tmp1))
!            !tmp1    = vector_exp(tmp1)
!            !write (6, *) "exponents"
!            !call print_vector(tmp1, size(tmp1))
!            rowsum  = sum(vector_exp(zT(:, i) - zmax))
!            zsum(i) = log(rowsum) + zmax
!        end do
!    end subroutine

!    function vector_exp(x) result(y)
!        real(r8k), intent(in) :: x(:)
!        real(r8k)             :: y(size(x))
!        integer(i4k)          :: i
!        do i = 1, size(x)
!            y(i) = exp(x(i))
!        end do
!    end function

    subroutine log_sum_exponent2d(nrow, ncol, z, zsum)
!
!   Given a DOUBLE PRECISION matrix z of size (n x p) - n rows and p columns
!   return
!       zsum(p) = log(sum_i(exp( z_{ij} )))
!   That is, the sum is over each row of z.
!
        integer(i4k), intent(in) :: nrow, ncol
        real(r8k), intent(in)    :: z(nrow, ncol)
        real(r8k), intent(out)   :: zsum(ncol) 
        real(r8k)                :: colmax(ncol), colsum
        integer(i4k)             :: j!, i
        real(r8k)                :: rtmp1
        colmax = maxval(z, 1)
        do j = 1, ncol
            !colmax = maxval(z(:, j))
            !colmax = z(1, j)
            !do i = 2, nrow
            !    if (z(i, j) .ge. colmax) then
            !        colmax = z(i, j)
            !    end if
            !end do
            rtmp1 = colmax(j)
            !colsum = d_zero
            !do i = 1, nrow
            !    colsum = colsum + exp(z(i, j) - rtmp1)
            !end do
            colsum = sum( exp( z(:,j) - rtmp1 ) )
            zsum(j) = log(colsum) + rtmp1
        end do
    end subroutine

    subroutine softmax(x, logbase, w)
!
!   Given DOUBLE PRECISION array a(k) and 
!         DOUBLE PRECISION variable smlb = log(b), 
!
!   calculate DOUBLE PRECISION array w(k):
!
!       w_k = b^(a_k) / sum(b^(a_k))
!   
!   Note if log(b) = 1
!        then b = exp(1) and w_k = exp(a_k) / sum(exp(a_k))
!        if log(b) = z
!        then b = exp(z) and w_k = exp(z * a_k) / sum(exp(z * a_k))
!
        real(r8k), intent(in)  :: x(:)
        real(r8k), intent(in)  :: logbase
        real(r8k), intent(out) :: w(size(x))
        real(r8k)              :: bx(size(x))
        real(r8k)              :: bxmax, wsum
        integer(i4k)           :: i
        do i = 1, size(x)
            bx(i) = x(i) * logbase
        end do
        bxmax = maxval(bx)
        do i = 1, size(x)
            w(i) = exp(bx(i) - bxmax)
        end do
        wsum = sum(w)
        do i = 1, size(x)
            w(i) = w(i) / wsum
        end do
    end subroutine softmax

    subroutine softmax_jacobian(w, logbase, jac)
        real(r8k), intent(in)  :: w(:)
        real(r8k), intent(in)  :: logbase
        real(r8k), intent(out) :: jac(size(w), size(w))
        integer(i4k)           :: i, j
        real(r8k)              :: wj
        do j = 1, size(w)
            wj = w(j)
            jac(j, j) = wj * (1 - wj) * logbase
            do i = 1, size(w)
                if (i /= j) then
                    jac(i, j) = - w(i) * wj * logbase
                end if
            end do
        end do
    end subroutine softmax_jacobian

    subroutine softmax_gradient(n, w, smlb, dfdw, dfda)
!   
!   Let w = softmax(a, smlb)
!   For any function f(w), this subroutine calculates
!   df/da given df/dw, w and smlb
!
        implicit none
        integer(i4k)  :: n
        real(r8k)     :: dfdw(n), dfda(n)
        real(r8k)     :: w(n)
        real(r8k)     :: smlb
        real(r8k)     :: ajac(n, n)
        integer(i4k)  :: i, j
        call softmax_jacobian(w, smlb, ajac)
        dfda = d_zero
        do j = 1, n
            do i = 1, n
                dfda(j) = dfda(j) + dfdw(i) * ajac(i, j)
            end do
        end do
    end subroutine

    subroutine duplicate_columns(x, ncol, y)
        integer(i4k), intent(in) :: ncol
        real(r8k), intent(in)    :: x(:)
        real(r8k)                :: y(size(x), ncol)
        integer(i4k)             :: i
        do i = 1, ncol
            y(:, i) = x 
        end do
    end subroutine

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
