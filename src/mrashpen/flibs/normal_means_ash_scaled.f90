module nmash_scaled
    use env_precision
    use global_parameters
    implicit none
!    private
!    public initialize, log_sum_exponent2d

!   ====================
!   Global variables in this module
!   ====================
!   yj  vector of length n | input data vector
!   s   scalar | standard deviation of the NM model
!   wk  vector of length k  | prior mixture proportions 
!   sk  vector of length k  | prior mixture standard deviations
!   dj  vector of length n | a scaling vector for s
    real(r8k), allocatable :: y(:), wk(:), sk(:), dj(:)
    integer, allocatable   :: nzwk_idx(:)
    real(r8k) :: stddev
    integer(i4k) :: ndim, ncomp

contains

    subroutine initialize(yinit, sinit, wkinit, skinit, djinit, n, k)
        integer(i4k), intent(in) :: n, k
        real(r8k), intent(in)    :: sinit
        real(r8k), intent(in)    :: yinit(n), djinit(n)
        real(r8k), intent(in)    :: wkinit(k), skinit(k)
        integer(i4k) :: i, nzwk_count
!       set the global variables of this module
        ndim   = n
        ncomp  = k
        stddev = sinit
        allocate (y(ndim), dj(ndim))
        allocate (wk(ncomp), sk(ncomp))
        do i = 1, ndim
            y(i) = yinit(i)
            dj(i) = djinit(i)
        end do
        nzwk_count = 0
        do i = 1, ncomp
            sk(i) = skinit(i)
            wk(i) = wkinit(i)
            if (wk(i) .ne. d_zero) then
                nzwk_count = nzwk_count + 1
            end if
        end do
        allocate (nzwk_idx(nzwk_count))
        do i = 1, ncomp
            if (wk(i) .ne. d_zero) then
                nzwk_idx(i) = 1
            else
                nzwk_idx(i) = 0
            end if
        end do
    end subroutine


!    function calculate_logML() result(logML)
!        real(r8k) :: logML(ndim)
!        real(r8k) :: logLjk(ndim, ncomp), t1(ndim)
!        integer(i4k) :: i
!
!        call calculate_logLjk(0, logLjk)
!        write (6, *) "logLjk =>"
!        call print_array2d(logLjk, ndim, ncomp)
!        write (6, *) "wk =>"
!        call print_vector(wk, ncomp)
!        t1 = log_sum_wkLjk(wk, logLjk, nzwk_idx)
!        write (6, *) "log(sum(wk * Ljk)) =>"
!        call print_vector(t1, ndim)
!        do i = 1, ndim
!            logML(i) = - d_half * log2pi + t1(i)
!        end do
!    end function


    function log_sum_wkLjk(wk, logLjk, nzwk_idx) result(sumwL)
        integer, intent(in)   :: nzwk_idx(:)
        real(r8k), intent(in) :: wk(:), logLjk(:,:)
        real(r8k)             :: sumwL(size(logLjk, 1))
        real(r8k), allocatable :: z(:, :)
        integer(i4k) :: nk, nn, nz, i, j, k
        nz = sum(nzwk_idx)
        allocate(z(ndim, nz))
        k = 0
        do j = 1, ncomp
            if (nzwk_idx(j) .eq. 1) then
                k = k + 1
                do i = 1, ndim
                    z(i, k) = logLjk(i, j) + log(wk(j))
                end do
            end if
        end do
        write (6, *) "Dimensions of z", ndim, nz
        write (6, *) "log(wk) + logLjk =>"
        call print_vector(z, nz)
        sumwL = log_sum_exponent2d(z)
    end function


    function log_sum_exponent2d(z) result(zsum)
!        integer{i4k), intent(in) :: nrow
        real(r8k), intent(in)  :: z(:,:)
        real(r8k)              :: zsum(size(z, 1))
        real(r8k), allocatable :: zmax(:)
        integer(i4k) :: i, j, nrow, ncol
        nrow = size(z, 1)
        ncol = size(z, 2)
        allocate(zmax(nrow))
        do i = 1, nrow
            zsum(i) = d_zero
            zmax(i) = d_zero
        end do
        do j = 1, ncol
            do i = 1, nrow
                if (z(i, j) .gt. zmax(i)) then
                    zmax(i) = z(i, j)
                end if
            end do
        end do
        do j = 1, ncol
            do i = 1, nrow
                zsum(i) = zsum(i) + exp(z(i, j) - zmax(i))
            end do
        end do
        do i = 1, nrow
            zsum(i) = log(zsum(i)) + zmax(i)
        end do
    end function


!    subroutine calculate_logLjk(order, logLjk)
!        integer(i4k), intent(in) :: order
!        real(r8k), intent(out)   :: logLjk(ndim, ncomp)
!!       local variables
!        integer(i4k) :: i, j
!        real(r8k)    :: s2, logs2, sk2, v2jk, t1, t2, a1
!
!        select case (order)
!            case (0)
!                a1 = d_one
!            case (1)
!                a1 = 3 * d_one
!            case (2)
!                a1 = 5 * d_one
!        end select
!
!        s2 = stddev * stddev
!        logs2 = log(s2)
!        do j = 1, ncomp
!            sk2 = sk(j) ** d_two
!            do i = 1, ndim
!                v2jk = sk2 + (d_one / dj(i))
!                t1   = logs2 + log(v2jk)
!                t2   = (y(i) * y(i)) / (v2jk * s2)
!                logLjk(i, j) = - d_half * (a1 * t1 + t2)
!            end do
!        end do
!    end subroutine calculate_logLjk

end module nmash_scaled

! subroutine a_plus_matB(a, b, m, n, res)
!     integer(i4k), intent(in) :: m, n
!     real(r8k), intent(in)    :: a
!     real(r8k), intent(in)    :: b(m, n)
!     real(r8k), intent(out)   :: res(m, n)
!     integer(i4k)             :: i, j
!     do j = 1, n
!         do i = 1, m
!             res(i, j) = b(i, j) + a
!         end do
!     end do
! end subroutine a_plus_matB
! 
