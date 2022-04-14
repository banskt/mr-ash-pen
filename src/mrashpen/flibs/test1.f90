program main
    use env_precision
    use global_parameters
    use futils
    use normal_means_ash_scaled
    use plr_mrash
    use lbfgsb_driver
    implicit none
    integer, parameter :: nsample = 5, ndim = 6, ncomp = 3
    real(r8k) :: X(nsample, ndim), X2(nsample, ndim)
    real(r8k) :: y(nsample), dj(ndim), b(ndim), djinv(ndim)
    real(r8k) :: wk(ncomp), sk(ncomp)
    real(r8k) :: s, s2
    integer(i4k) :: i, j
    real(r8k) :: obj, bgrad(ndim), wgrad(ncomp), agrad(ncomp), s2grad
    real(r8k) :: t1, t2
    real(r8k) :: bopt(ndim), wopt(ncomp), s2opt, objopt
    real(r8k), allocatable :: gradopt(:)
    integer(i4k) :: nfev, niter
    character(len=60) :: task
!
!       NMash variables
    integer, parameter :: p = 6, k = 3
    real(r8k), dimension(p)    :: lml 
    real(r8k), dimension(p)    :: lml_bd,  lml_bd_bd
    real(r8k), dimension(k, p) :: lml_wd,  lml_bd_wd
    real(r8k), dimension(p)    :: lml_s2d, lml_bd_s2d

    real(r8k) :: ak(ncomp), wkmod(ncomp), akjac(ncomp, ncomp)
    real(r8k) :: smlogbase

    integer(i4k) :: nparams


    allocate(gradopt(ndim + ncomp + 1))


    s = 0.9d0
    s2 = s ** d_two
    y = [3.5d0, 4.5d0, 1.2d0, 6.5d0, 2.8d0]
    DATA               X/                    &
       8.79, 6.11,-9.15, 9.57,-3.49, 9.84,   &
       9.93, 6.91,-7.93, 1.64, 4.02, 0.15,   &
       9.83, 5.04, 4.86, 8.83, 9.80,-8.99,   &
       5.45,-0.27, 4.85, 0.74,10.00,-6.02,   &
       3.16, 7.98, 3.01, 5.80, 4.27,-5.31    &
                        /
    b = [1.21, 2.32, 0.01, 0.03, 0.11, 3.12]
    sk = [0.1d0, 0.5d0, 0.9d0]
    wk = [0.5d0, 0.25d0, 0.25d0]

    smlogbase = 1.0d0 
    do i = 1, ncomp
        ak(i) = log(wk(i)) / smlogbase
    end do
    call softmax(ak, smlogbase, wkmod)
!    wkmod = softmax(ak, smlogbase)

!    call deepcopy_matrix(X, Xcopy1, nsample, ndim)
!    call deepcopy_matrix(X, Xcopy2, nsample, ndim)
!    call dgemm('T', 'N', ndim, ndim, nsample, d_one, Xcopy1, nsample, Xcopy2, nsample, d_zero, XTX, ndim)

    X2 = X ** d_two
    dj = sum(X2, 1)
    do i = 1, ndim
        djinv(i) = d_one / dj(i)
    end do

    obj = 0
    bgrad = d_zero
    wgrad = d_zero
    s2grad = 0
    write (6, *) "exponent of -1020"
    t1 = -1020.d0
    t2 = exp(t1)
    write (6, *) t2
    write (6, *) "Enter plr_mrash subroutine"
    call plr_obj_grad_shrinkop(nsample, ndim, X, y, b, s2, ncomp, wk, sk, djinv, obj, bgrad, wgrad, s2grad)
    write (6, *) "Objective = ", obj
    write (6, *) "Returned"
    call softmax_jacobian(wk, smlogbase, akjac)
!    akjac = softmax_jacobian(wk, smlogbase)
    agrad = d_zero
    do j = 1, ncomp
        do i = 1, ncomp
            agrad(j) = agrad(j) + wgrad(i) * akjac(i, j) 
        end do
    end do

    write (6, *) "Enter normal_means_ash_lml subroutine" 
    call normal_means_ash_lml(p, k, b, s2, wk, sk, djinv,                           &
                              lml, lml_bd, lml_wd, lml_s2d,                         &
                              lml_bd_bd, lml_bd_wd, lml_bd_s2d)
    write (6, *) "Returned"


!    nparams = ndim + ncomp + 1
!    call min_plr_shrinkop(nsample, ndim, X, y, ncomp, b, wk, s2, sk,           & 
!                    nparams, .TRUE., .TRUE., .TRUE.,                           &
!                    d_one, 10, 1, 1.0d+7, 1.0d-5, 10, 1000,                    &
!                    bopt, wopt, s2opt, objopt, gradopt, nfev, niter, task)


    write (6, *) "Input data =>"
    write (6, *) "y"
    call print_vector(y, nsample)
    write (6, *) "X"
    call print_array2d(X, nsample, ndim)
    write (6, *) "ak"
    call print_vector(ak, ncomp)
    write (6, *) "wk"
    call print_vector(wkmod, ncomp)
    write (6, *) "Output data =>"
    write (6, *) "log ML"
    call print_vector(lml, p)
    write (6, *) "Objective = ", obj
    write (6, *) "logML_deriv"
    call print_vector(lml_bd, p)
    write (6, *) "bgrad"
    call print_vector(bgrad, ndim)
    write (6, *) "wgrad"
    call print_vector(wgrad, ncomp)
    write (6, *) "Softmax jacobian"
    call print_array2d(akjac, ncomp, ncomp)
    write (6, *) "agrad"
    call print_vector(agrad, ncomp)
    write (6, *) "s2grad"
    write (6, *) s2grad
end program 
