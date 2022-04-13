module normal_means_ash_scaled
    use env_precision
    use global_parameters
    use futils, only: log_sum_exponent2d
    implicit none

contains

    subroutine normal_means_ash_lml(ndim, ncomp, y, s2, wk, sk, djinv,           &
                                    lml, lml_bd, lml_wd, lml_s2d,                &
                                    lml_bd_bd, lml_bd_wd, lml_bd_s2d)
        integer(i4k), intent(in) :: ndim, ncomp
        real(r8k), intent(in)    :: y(ndim), djinv(ndim)
        real(r8k), intent(in)    :: s2
        real(r8k), intent(in)    :: wk(ncomp), sk(ncomp)

        real(r8k), dimension(ndim), intent(out) :: lml
        real(r8k), dimension(ndim), intent(out) :: lml_bd,    lml_s2d
        real(r8k), dimension(ndim), intent(out) :: lml_bd_bd, lml_bd_s2d
        real(r8k), dimension(ndim, ncomp), intent(out) :: lml_wd
        real(r8k), dimension(ndim, ncomp), intent(out) :: lml_bd_wd

!       local variables
        real(r8k), dimension(ncomp, ndim) :: lkp0, lkp1, lkp2
        real(r8k), dimension(ncomp, ndim) :: v2kp, logv2kp
        real(r8k), dimension(ncomp, ndim) :: lml_wdT, lml_bd_wdT
        real(r8k), dimension(ndim)        :: lsum_wl0, lsum_wl1, lsum_wl2
        real(r8k), dimension(ndim)        :: lsum_wl1v2, lsum_wl2v2
        real(r8k), dimension(ndim)        :: lml_bd_over_y
        real(r8k), dimension(ndim)        :: y2
        real(r8k), dimension(ncomp)       :: sk2, logwk
        integer, dimension(ncomp)         :: nzwk_idx
        integer                           :: i, j
        real(r8k)                         :: rtmp1, rtmp2

        y2 = y ** d_two
        sk2 = sk ** d_two

        do j = 1, ndim
            do i = 1, ncomp
                rtmp1 = sk2(i) + djinv(j)
                v2kp(i, j) = rtmp1
                logv2kp(i, j) = log(rtmp1)
            end do
        end do

        do i = 1, ncomp
            if (wk(i) .gt. d_zero) then
                nzwk_idx(i) = 1
                logwk(i)    = log(wk(i))
            else
                nzwk_idx(i) = 0
                logwk(i)    = -huge(d_zero)
            end if
        end do

!        lkp0 = d_zero
!        lkp1 = d_zero
!        lkp2 = d_zero
        call calculate_logLkp(ndim, ncomp, y, s2, v2kp, logv2kp,                  &
                              lkp0, lkp1, lkp2)
!       sum over wk to obtain vectors of size ndim
        lsum_wl0   = log_sum_wkLkp( logwk, lkp0, nzwk_idx )
        lsum_wl1   = log_sum_wkLkp( logwk, lkp1, nzwk_idx )
        lsum_wl2   = log_sum_wkLkp( logwk, lkp2, nzwk_idx )
        lsum_wl1v2 = log_sum_wkLkp( logwk, lkp1 + logv2kp, nzwk_idx )
        lsum_wl2v2 = log_sum_wkLkp( logwk, lkp2 + logv2kp, nzwk_idx )

!       ========================
!       Calculate L (log marginal likelihood)
!       ========================
        lml = - d_half * log2pi + lsum_wl0

!       ========================
!       Calculate derivatives of L with respect to b, w and s2
!       ========================
        do j = 1, ndim
            rtmp1             = exp(lsum_wl1(j) - lsum_wl0(j))
            lml_bd_over_y(j)  = - rtmp1
            lml_bd(j)         = y(j) * lml_bd_over_y(j)
            do i = 1, ncomp
                lml_wdT(i, j) = exp(-d_half * log2pi + lkp0(i, j) - lml(j))
            end do
            lml_s2d(j)        = ((y2(j) / s2) * rtmp1 - exp(lsum_wl1v2(j) - lsum_wl0(j))) * d_half
        end do
        lml_wd  = transpose(lml_wdT)

!       ========================
!       Calculate derivatives of L' with respect to b, w and s2
!       ========================
        do j = 1, ndim
            rtmp1                 = exp(lsum_wl2(j) - lsum_wl0(j))
            lml_bd_bd(j)          = lml_bd_over_y(j) + y2(j) * rtmp1 - (lml_bd(j) * lml_bd(j))
            do i = 1, ncomp
                lml_bd_wdT(i, j)  = - lml_wdT(i, j) * (y(j) * exp(lkp1(i, j) - lkp0(i, j)) + lml_bd(j))
            end do
            rtmp2                 = - (y2(j) / s2) * rtmp1 + 3.0d0 * exp(lsum_wl2v2(j) - lsum_wl0(j))
            lml_bd_s2d(j)         = (d_half * y(j) * rtmp2) - (lml_bd(j) * lml_s2d(j))
        end do
        lml_bd_wd = transpose(lml_bd_wdT)
    end subroutine normal_means_ash_lml

    subroutine calculate_logLkp(p, k, y, sigma2, v2kp, logv2kp,                  &
                                logLkp0, logLkp1, logLkp2)
        integer(i4k), intent(in) :: p, k
        real(r8k), intent(in)    :: y(p)
        real(r8k), intent(in)    :: sigma2
        real(r8k), intent(in)    :: v2kp(k, p), logv2kp(k, p)
        real(r8k), dimension(k, p), intent(out) :: logLkp0, logLkp1, logLkp2
!       local variables
        integer(i4k) :: j
        real(r8k)    :: a0, a1, a2
        real(r8k)    :: logs2!, t1, t2
        real(r8k)    :: arr1(k), arr2(k)
        real(r8k)    :: y2_by_s2(p)

        logs2 = log(sigma2)
        a0 = d_one
        a1 = 3 * d_one
        a2 = 5 * d_one
        !do i = 1, p
        y2_by_s2 = (y ** d_two) / sigma2
        !end do
        do j = 1, p
            arr1 = logs2 + logv2kp(:, j)
            arr2 = y2_by_s2(j) / v2kp(:, j)
            logLkp0(:, j) = - d_half * (a0 * arr1 + arr2)
            logLkp1(:, j) = - d_half * (a1 * arr1 + arr2)
            logLkp2(:, j) = - d_half * (a2 * arr1 + arr2)
        end do
    end subroutine calculate_logLkp

    function log_sum_wkLkp(logwk, logLkp, nzwk_idx) result(sumwL)
        integer, intent(in)    :: nzwk_idx(:)
        real(r8k), intent(in)  :: logwk(:), logLkp(:,:)
        real(r8k)              :: sumwL(size(logLkp, 2))
        real(r8k), allocatable :: z(:, :)
        integer(i4k) :: nk, np, nz, i, j, k
!
        np = size(logLkp, 2)
        nk = size(logwk)
        nz = sum(nzwk_idx)
!
        if( allocated(z) )  deallocate( z ) 
        allocate(z(nz, np))

        do j = 1, np
!            z(:, j) = logLkp(:, j) + logwk
            k = 0
            do i = 1, nk
                if (nzwk_idx(i) .eq. 1) then
                    k = k + 1
                    z(k, j) = logLkp(k, j) + logwk(k)
                end if
            end do
        end do
        call log_sum_exponent2d(nz, np, z, sumwL)
    end function

end module normal_means_ash_scaled
