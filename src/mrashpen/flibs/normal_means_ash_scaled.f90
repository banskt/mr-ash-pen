module normal_means_ash_scaled
    use env_precision
    use global_parameters
    use futils, only: log_sum_exponent2d!, print_vector
    implicit none

contains

    subroutine normal_means_ash_lml(ndim, ncomp, y, s2, wk, sk, djinv,           &
                                    lml, lml_bd, lml_wdT, lml_s2d,               &
                                    lml_bd_bd, lml_bd_wdT, lml_bd_s2d)
        integer(i4k), intent(in) :: ndim, ncomp
        real(r8k), intent(in)    :: y(ndim), djinv(ndim)
        real(r8k), intent(in)    :: s2
        real(r8k), intent(in)    :: wk(ncomp), sk(ncomp)

        real(r8k), dimension(ndim), intent(out) :: lml
        real(r8k), dimension(ndim), intent(out) :: lml_bd,    lml_s2d
        real(r8k), dimension(ndim), intent(out) :: lml_bd_bd, lml_bd_s2d
        real(r8k), dimension(ncomp, ndim), intent(out) :: lml_wdT
        real(r8k), dimension(ncomp, ndim), intent(out) :: lml_bd_wdT

!       local variables
        real(r8k), dimension(ncomp, ndim) :: lkp0, lkp1, lkp2
        real(r8k), dimension(ncomp, ndim) :: v2kp, logv2kp
!        real(r8k), dimension(ncomp, ndim) :: lml_wdT, lml_bd_wdT
        real(r8k), dimension(ndim)        :: lsum_wl0, lsum_wl1, lsum_wl2
        real(r8k), dimension(ndim)        :: lsum_wl1v2, lsum_wl2v2
        !real(r8k), dimension(ndim)        :: lml_bd_over_y
        real(r8k), dimension(ndim)        :: y2
        real(r8k), dimension(ncomp)       :: sk2, logwk
        integer, allocatable              :: nzwk_idx(:)
        integer                           :: i, j, nzk
        real(r8k)                         :: arr1(ncomp)
        real(r8k) :: e_wl1_wl0, e_wl2_wl0, e_wl1v2_wl0, e_wl2v2_wl0, y2_by_s2
        real(r8k) :: e_lkp1_lkp0(ncomp)
        real(r8k) :: rtmp1, lml_bdj
!
!       ========================
!       Precalculate variables
!       ========================
        y2 = y ** d_two
        sk2 = sk ** d_two
!
        nzk = 0
        do i = 1, ncomp
            if (wk(i) .gt. d_zero) then
                nzk = nzk + 1
            end if
        end do
        if( allocated(nzwk_idx) )  deallocate( nzwk_idx )
        allocate(nzwk_idx(nzk))
!
        j = 0
        do i = 1, ncomp
            if (wk(i) .gt. d_zero) then
                j = j + 1
                nzwk_idx(j) = i
                logwk(i) = log(wk(i))
            else
                logwk(i) = - huge(d_zero)
            end if
        end do
!
        do j = 1, ndim
            arr1 = sk2(:) + djinv(j)
            v2kp(:, j) = arr1
            logv2kp(:, j) = log(arr1)
        end do
!        lkp0 = d_zero
!        lkp1 = d_zero
!        lkp2 = d_zero
        call calculate_logLkp(ndim, ncomp, y2, s2, v2kp, logv2kp,                  &
                              lkp0, lkp1, lkp2)
!       sum over wk to obtain vectors of size ndim
        call log_sum_wkLkp( ncomp, ndim, nzk, logwk, lkp0,           nzwk_idx, lsum_wl0 )
        call log_sum_wkLkp( ncomp, ndim, nzk, logwk, lkp1,           nzwk_idx, lsum_wl1 )
        call log_sum_wkLkp( ncomp, ndim, nzk, logwk, lkp2,           nzwk_idx, lsum_wl2 )
        call log_sum_wkLkp( ncomp, ndim, nzk, logwk, lkp1 + logv2kp, nzwk_idx, lsum_wl1v2 )
        call log_sum_wkLkp( ncomp, ndim, nzk, logwk, lkp2 + logv2kp, nzwk_idx, lsum_wl2v2 )
!
!       ========================
!       Calculate L (log marginal likelihood)
!       ========================
        lml = - d_half * log2pi + lsum_wl0
!
!       ========================
!       Calculate derivatives of L with respect to b, w and s2
!       ========================
!       e_wl1_wl0   = exp( lsum_wl1 - lsum_wl0 )
!       e_wl2_wl0   = exp( lsum_wl2 - lsum_wl0 )
!       e_lkp1_lkp0 = exp( lkp1 - lkp0 )
!       e_wl1v2_wl0 = exp( lsum_wl1v2 - lsum_wl0 )
!       e_wl2v2_wl0 = exp( lsum_wl2v2 - lsum_wl0 )
!       y2_by_s2    = y2 / s2
        do j = 1, ndim
            e_wl1_wl0   = exp( lsum_wl1(j) - lsum_wl0(j) )
            e_wl2_wl0   = exp( lsum_wl2(j) - lsum_wl0(j) )
            e_wl1v2_wl0 = exp( lsum_wl1v2(j) - lsum_wl0(j) )
            e_wl2v2_wl0 = exp( lsum_wl2v2(j) - lsum_wl0(j) )
            y2_by_s2    = y2(j) / s2

            lml_bd(j)             = - y(j) * e_wl1_wl0
            !do i = 1, ncomp
            !    lml_wdT(i, j)     = exp(-d_half * log2pi + lkp0(i, j) - lml(j))
            !end do
            lml_wdT(:, j)         = exp(-d_half * log2pi + lkp0(:, j) - lml(j))
            lml_s2d(j)            = ( (y2_by_s2 * e_wl1_wl0)  - e_wl1v2_wl0 ) * d_half
!       ========================
!       Calculate derivatives of L' with respect to b, w and s2
!       ========================
            lml_bdj               = lml_bd(j)
            lml_bd_bd(j)          =  - e_wl1_wl0 + y2(j) * e_wl2_wl0 - (lml_bdj * lml_bdj)
            !do i = 1, ncomp
            !    e_lkp1_lkp0       = exp( lkp1(i,j) - lkp0(i,j) )
            !    lml_bd_wdT(i, j)  = - lml_wdT(i, j) * (y(j) * e_lkp1_lkp0 + lml_bdj)
            !end do
            e_lkp1_lkp0           = exp( lkp1(:,j) - lkp0(:,j) )
            lml_bd_wdT(:, j)      = - lml_wdT(:, j) * (y(j) * e_lkp1_lkp0 + lml_bdj)
            rtmp1                 = - y2_by_s2 * e_wl2_wl0 + 3.0d0 * e_wl2v2_wl0
            lml_bd_s2d(j)         = (d_half * y(j) * rtmp1) - (lml_bdj * lml_s2d(j))
        end do
! !       ========================
! !       Calculate derivatives of L with respect to b, w and s2
! !       ========================
!         do j = 1, ndim
!             rtmp1             = exp(lsum_wl1(j) - lsum_wl0(j))
!             lml_bd_over_y(j)  = - rtmp1
!             lml_bd(j)         = y(j) * lml_bd_over_y(j)
!             do i = 1, ncomp
!                 lml_wdT(i, j) = exp(-d_half * log2pi + lkp0(i, j) - lml(j))
!             end do
!             lml_s2d(j)        = ((y2(j) / s2) * rtmp1 - exp(lsum_wl1v2(j) - lsum_wl0(j))) * d_half
!         end do
!         lml_wd  = transpose(lml_wdT)
! 
! !       ========================
! !       Calculate derivatives of L' with respect to b, w and s2
! !       ========================
!         do j = 1, ndim
!             rtmp1                 = exp(lsum_wl2(j) - lsum_wl0(j))
!             lml_bd_bd(j)          = lml_bd_over_y(j) + y2(j) * rtmp1 - (lml_bd(j) * lml_bd(j))
!             do i = 1, ncomp
!                 lml_bd_wdT(i, j)  = - lml_wdT(i, j) * (y(j) * exp(lkp1(i, j) - lkp0(i, j)) + lml_bd(j))
!             end do
!             rtmp2                 = - (y2(j) / s2) * rtmp1 + 3.0d0 * exp(lsum_wl2v2(j) - lsum_wl0(j))
!             lml_bd_s2d(j)         = (d_half * y(j) * rtmp2) - (lml_bd(j) * lml_s2d(j))
!         end do
!         lml_bd_wd = transpose(lml_bd_wdT)
    end subroutine normal_means_ash_lml
!
!
    subroutine calculate_logLkp(p, k, y2, sigma2, v2kp, logv2kp,                    &
                                logLkp0, logLkp1, logLkp2)
        integer(i4k), intent(in) :: p, k
        real(r8k), intent(in)    :: y2(p)
        real(r8k), intent(in)    :: sigma2
        real(r8k), intent(in)    :: v2kp(k, p), logv2kp(k, p)
        real(r8k), dimension(k, p), intent(out) :: logLkp0, logLkp1, logLkp2
!       local variables
        integer(i4k) :: j
        real(r8k)    :: logs2
        real(r8k)    :: arr1(k), arr2(k)
        real(r8k)    :: y2_by_s2(p)

        logs2 = log(sigma2)
        y2_by_s2 = y2 / sigma2
        do j = 1, p
            arr1 = logs2 + logv2kp(:, j)
            arr2 = y2_by_s2(j) / v2kp(:, j)
            logLkp0(:, j) = - d_half * (arr1 + arr2)
            logLkp1(:, j) = logLkp0(:, j) - arr1
            logLkp2(:, j) = logLkp1(:, j) - arr1
        end do
    end subroutine calculate_logLkp

    subroutine log_sum_wkLkp(nk, np, nzk, logwk, logLkp, nzwk_idx, sumwL)
        integer, intent(in)    :: nk, np, nzk
        integer, intent(in)    :: nzwk_idx(nzk)
        real(r8k), intent(in)  :: logwk(nk), logLkp(nk, np)
        real(r8k)              :: sumwL(np)
        real(r8k), allocatable :: z(:, :)
        integer(i4k) :: i, j, k
!
        if( allocated(z) )  deallocate( z ) 
        allocate(z(nzk, np))

        do j = 1, np
!            z(:, j) = logLkp(:, j) + logwk
            do i = 1, nzk
                k = nzwk_idx(i)
                z(i, j) = logLkp(k, j) + logwk(k)
            end do
        end do
        call log_sum_exponent2d(nzk, np, z, sumwL)
    end subroutine log_sum_wkLkp

end module normal_means_ash_scaled
