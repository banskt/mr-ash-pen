module get_data
    use env_precision
    use global_parameters
    use futils
    implicit none

contains

!    subroutine get_mixgauss_scale(k, b, sk)
!        integer(i4k), intent(in) :: k
!        real(r8k), intent(in)    :: b
!        real(r8k), intent(out)   :: sk(k)
!        integer(i4k)             :: i
!        do i = 1, k
!            sk(i) = b ** ((i - d_one) / k) - d_one
!        end do
!    end subroutine
!
!    subroutine get_mixgauss_weight(k, sprs, wk)
!        integer(i4k), intent(in) :: k
!        real(r8k), intent(in)    :: sprs
!        real(r8k), intent(out)   :: wk(k)
!        integer(i4k)             :: i
!        wk = d_zero
!        if (sprs .ne. d_zero) then
!            wk(1) = sprs
!        else
!            wk(1) = d_one / k
!        end if
!        do i = 2, k - 1
!            wk(i) = (1 - wk(1)) / (k - d_one)
!        end do
!        wk(k) = d_one - sum(wk(1:k-1))
!    end subroutine

    subroutine get_predictors(n, p, x)
        integer(i4k), intent(in) :: n, p
        real(r8k), intent(out)   :: x(n, p)
        integer(i4k)             :: i
        real(r8k)                :: xtmp(n * p)
        integer(i4k)             :: iseed(4)
        iseed(1) = 0
        iseed(2) = 0
        iseed(3) = 0
        iseed(4) = 1
        call random_normal(n * p, iseed, xtmp)
        do i = 1, p
            x(:, i) = xtmp( (i-1)*n+1 : i*n )
        end do
        call center_and_scale(n, p, x)
    end subroutine

    subroutine get_sparse_coef(p, pc, b)
        integer(i4k), intent(in) :: p, pc
        real(r8k), intent(out)   :: b(p)
        integer(i4k)             :: iseed(4)
        iseed(1) = 1
        iseed(2) = 1
        iseed(3) = 1
        iseed(4) = 3
        b = d_zero
        call random_normal(pc, iseed, b(1:pc))
    end subroutine

    subroutine get_responses(n, p, x, b, s2, y)
        integer(i4k), intent(in) :: n, p
        real(r8k), intent(in)    :: x(n, p), b(p), s2
        real(r8k), intent(out)   :: y(n)
        real(r8k)                :: err(n), xdotb(n)
        integer(i4k)             :: iseed(4)
        iseed(1) = 3
        iseed(2) = 3
        iseed(3) = 3
        iseed(4) = 5
        call random_normal(n, iseed, err) 
        err = err * sqrt(s2)
        !xdotb = d_zero
        call dgemv('N', n, p, d_one, x, n, b, 1, d_zero, xdotb, 1)
        y = xdotb + err
    end subroutine

    subroutine center_and_scale(n, p, x)
        integer(i4k) :: n, p
        real(r8k)    :: x(n, p)
        integer(i4k) :: i
        real(r8k)    :: y(n), ysum, y2sum, yvar, ymean
        do i = 1, p
            y = x(:, i)
            ysum = sum(y)
            y2sum = sum(y ** d_two)
            yvar = (y2sum - ysum * ysum / n) / n
            x(:, i) = y / sqrt(yvar)
        end do
        do i = 1, p
            y = x(:, i)
            ysum = sum(y)
            ymean = ysum / n
            x(:, i) = y - ymean
        end do
    end subroutine

    subroutine random_normal(n, iseed, x)
        integer(i4k), intent(in) :: n
        integer(i4k), intent(in) :: iseed(4)
        real(r8k), intent(out)  :: x(n)
        !x = d_zero
        call dlarnv( 3, iseed, n, x)
    end subroutine

end module
