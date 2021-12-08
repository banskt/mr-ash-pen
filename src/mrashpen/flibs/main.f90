program main
    use env_precision
    use global_parameters
    use futils
    use normal_means_ash_scaled
    use plr_mrash
    implicit none
    integer, parameter :: nsample = 5, ndim = 6, ncomp = 3
    real(r8k) :: X(nsample, ndim), X2(nsample, ndim)
    real(r8k) :: y(nsample), dj(ndim), b(ndim), djinv(ndim)
    real(r8k) :: wk(ncomp), sk(ncomp)
    real(r8k) :: s
    integer(i4k) :: i
    real(r8k) :: obj, bgrad(ndim), wgrad(ncomp), s2grad
    real(r8k) :: t1, t2

!
!       NMash variables
        integer, parameter :: p = 6, k = 3
        real(r8k), dimension(p)    :: lml 
        real(r8k), dimension(p)    :: lml_bd,  lml_bd_bd
        real(r8k), dimension(p, k) :: lml_wd,  lml_bd_wd
        real(r8k), dimension(p)    :: lml_s2d, lml_bd_s2d


    s = 0.9d0
    y = [3.5d0, 4.5d0, 1.2d0, 6.5d0, 2.8d0]
    DATA               X/                    &
       8.79, 6.11,-9.15, 9.57,-3.49, 9.84,   &
       9.93, 6.91,-7.93, 1.64, 4.02, 0.15,   &
       9.83, 5.04, 4.86, 8.83, 9.80,-8.99,   &
       5.45,-0.27, 4.85, 0.74,10.00,-6.02,   &
       3.16, 7.98, 3.01, 5.80, 4.27,-5.31    &
                        /
    b = [1.21, 2.32, 0.01, 0.03, 0.11, 3.12]
    sk = [0.d0, 0.1d0, 0.9d0]
    wk = [0.5d0, 0.25d0, 0.25d0]

!    call deepcopy_matrix(X, Xcopy1, nsample, ndim)
!    call deepcopy_matrix(X, Xcopy2, nsample, ndim)
!    call fill_real_matrix(XTX, d_zero)
!    call dgemm('T', 'N', ndim, ndim, nsample, d_one, Xcopy1, nsample, Xcopy2, nsample, d_zero, XTX, ndim)

    X2 = X ** d_two
    dj = sum(X2, 1)
    do i = 1, ndim
        djinv(i) = d_one / dj(i)
    end do

    obj = 0
    call fill_real_vector(bgrad, d_zero)
    call fill_real_vector(wgrad, d_zero)
    s2grad = 0
    write (6, *) "exponent of -1020"
    t1 = -1020.d0
    t2 = exp(t1)
    write (6, *) t2
    write (6, *) "Enter plr_mrash subroutine"
    call objective_gradients(nsample, ndim, X, y, b, s, ncomp, wk, sk, djinv, obj, bgrad, wgrad, s2grad)
    write (6, *) "Returned"

!       ========================
!       Use normal means model
!       ========================
!       Initialize
        call fill_real_vector(lml, d_zero)
        call fill_real_vector(lml_bd, d_zero)
        call fill_real_matrix(lml_wd, d_zero)
        call fill_real_vector(lml_s2d, d_zero)
        call fill_real_vector(lml_bd_bd, d_zero)
        call fill_real_matrix(lml_bd_wd, d_zero)
        call fill_real_vector(lml_bd_s2d, d_zero)
        call normal_means_ash_lml(p, k, b, s, wk, sk, djinv,                    &
                                  lml, lml_bd, lml_wd, lml_s2d,                      &
                                  lml_bd_bd, lml_bd_wd, lml_bd_s2d)


    write (6, *) "Input data =>"
    call print_vector(y, nsample)
    call print_array2d(X, nsample, ndim)
    write (6, *) "log ML"
    call print_vector(lml, p)
    write (6, *) "Objective = ", obj
    write (6, *) "bgrad =>"
    call print_vector(bgrad, ndim)
    write (6, *) "wgrad =>"
    call print_vector(wgrad, ncomp)
    write (6, *) "s2grad =>"
    write (6, *) s2grad
end program 
