module normal_means_ash_scaled
    use env_precision
    use global_parameters
    use futils
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

contains

    subroutine normal_means_ash_lml(ndim, ncomp, y, stddev, wk, sk, djinv,       &
                                    lml, lml_bd, lml_wd, lml_s2d,                &
                                    lml_bd_bd, lml_bd_wd, lml_bd_s2d)
        integer(i4k), intent(in) :: ndim, ncomp
        real(r8k), intent(in)    :: y(ndim), djinv(ndim)
        real(r8k), intent(in)    :: stddev
        real(r8k), intent(in)    :: wk(ncomp), sk(ncomp)

        real(r8k), dimension(ndim), intent(out) :: lml
        real(r8k), dimension(ndim), intent(out) :: lml_bd,    lml_s2d
        real(r8k), dimension(ndim), intent(out) :: lml_bd_bd, lml_bd_s2d
        real(r8k), dimension(ndim, ncomp), intent(out) :: lml_wd
        real(r8k), dimension(ndim, ncomp), intent(out) :: lml_bd_wd

!       local variables
        real(r8k), dimension(ndim, ncomp) :: ljk0, ljk1, ljk2
        real(r8k), dimension(ndim, ncomp) :: v2pk, logv2pk
        real(r8k), dimension(ndim, ncomp) :: lml_krep, y_krep, lml_bd_krep
        real(r8k), dimension(ndim)        :: lsum_wl0, lsum_wl1, lsum_wl2
        real(r8k), dimension(ndim)        :: lsum_wl1v2, lsum_wl2v2
        real(r8k), dimension(ndim)        :: lml_bd_over_y, vec1
        real(r8k), dimension(ndim)        :: y2
        real(r8k)                         :: s2, sk2
        integer, dimension(ncomp)         :: nzwk_idx
        integer                           :: i, j

        s2 = stddev ** d_two
        y2 = y ** d_two

        call fill_real_matrix(v2pk, d_zero)
        do j = 1, ncomp
            sk2 = sk(j) ** 2
            do i = 1, ndim
                v2pk(i, j) = sk2 + djinv(i)
            end do
        end do
        nzwk_idx = get_nonzero_index_vector(wk)

        call fill_real_matrix(ljk0, d_zero)
        call fill_real_matrix(ljk1, d_zero)
        call fill_real_matrix(ljk2, d_zero)
        call calculate_logLjk(ndim, ncomp, y, s2, v2pk,                          &
                              ljk0, ljk1, ljk2)
!       sum over wk to obtain vectors of size ndim
        lsum_wl0   = log_sum_wkLjk(wk, ljk0, nzwk_idx)
        lsum_wl1   = log_sum_wkLjk(wk, ljk1, nzwk_idx)
        lsum_wl2   = log_sum_wkLjk(wk, ljk2, nzwk_idx)
        logv2pk    = log(v2pk)
        lsum_wl1v2 = log_sum_wkLjk(wk, ljk1 + logv2pk, nzwk_idx)
        lsum_wl2v2 = log_sum_wkLjk(wk, ljk2 + logv2pk, nzwk_idx)

!       ========================
!       Calculate L (log marginal likelihood)
!       ========================
        lml = - d_half * log2pi + lsum_wl0

!       ========================
!       Calculate derivatives of L with respect to b, w and s2
!       ========================
        lml_krep      = duplicate_columns(lml, ncomp)
        lml_bd_over_y = - exp(lsum_wl1 - lsum_wl0)
        lml_bd        = y * lml_bd_over_y
        lml_wd        = exp(- d_half * log2pi + ljk0 - lml_krep)
        lml_s2d       = ((y2 / s2) * exp(lsum_wl1   - lsum_wl0)                  &
                                   - exp(lsum_wl1v2 - lsum_wl0)) * d_half

!       ========================
!       Calculate derivatives of L' with respect to b, w and s2
!       ========================
        y_krep        = duplicate_columns(y, ncomp)
        lml_bd_krep   = duplicate_columns(lml_bd, ncomp)
        lml_bd_bd     = lml_bd_over_y + y2 * exp(lsum_wl2 - lsum_wl0) - (lml_bd * lml_bd)
        lml_bd_wd     = - lml_wd * (y_krep * exp(ljk1 - ljk0) + lml_bd_krep)
        vec1          = - (y2 / s2) * exp(lsum_wl2   - lsum_wl0)                 &
                                + 3 * exp(lsum_wl2v2 - lsum_wl0)
        lml_bd_s2d    = (d_half * y * vec1) - (lml_bd * lml_s2d)
    end subroutine normal_means_ash_lml

    subroutine calculate_logLjk(p, k, y, sigma2, v2pk,                           &
                                logLjk0, logLjk1, logLjk2)
        integer(i4k), intent(in) :: p, k
        real(r8k), intent(in)    :: y(p)
        real(r8k), intent(in)    :: sigma2
        real(r8k), intent(in)    :: v2pk(p, k)
        real(r8k), dimension(p, k), intent(out) :: logLjk0, logLjk1, logLjk2
!       local variables
        integer(i4k) :: i, j
        real(r8k)    :: a0, a1, a2
        real(r8k)    :: logs2, t1, t2

        logs2 = log(sigma2)
        a0 = d_one
        a1 = 3 * d_one
        a2 = 5 * d_one
        do j = 1, k
            do i = 1, p
                t1   = logs2 + log(v2pk(i, j))
                t2   = (y(i) * y(i)) / (v2pk(i, j) * sigma2)
                logLjk0(i, j) = - d_half * (a0 * t1 + t2)
                logLjk1(i, j) = - d_half * (a1 * t1 + t2)
                logLjk2(i, j) = - d_half * (a2 * t1 + t2)
            end do
        end do
    end subroutine calculate_logLjk


    function log_sum_wkLjk(wk, logLjk, nzwk_idx) result(sumwL)
        integer, intent(in)    :: nzwk_idx(:)
        real(r8k), intent(in)  :: wk(:), logLjk(:,:)
        real(r8k)              :: sumwL(size(logLjk, 1))
        real(r8k), allocatable :: z(:, :)
        integer(i4k) :: nk, np, nz, j, k
        np = size(logLjk, 1)
        nk = size(wk)
        nz = sum(nzwk_idx)
        allocate(z(np, nz))
        k = 0
        do j = 1, nk
            if (nzwk_idx(j) .eq. 1) then
                k = k + 1
                z(:, k) = logLjk(:, j) + log(wk(j))
            end if
        end do
!        write (6, *) "Dimensions of z", np, nz
!        write (6, *) "log(wk) + logLjk =>"
!        call print_vector(z, nz)
        sumwL = log_sum_exponent2d(z)
    end function

end module normal_means_ash_scaled
