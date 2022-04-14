program main
    use env_precision
    use global_parameters
    use futils
!    use get_data
    use normal_means_ash_scaled
    use plr_mrash
    implicit none
    integer, parameter :: nsample = 500, ndim = 10000, ncomp = 20, ncausal = 10
    real(r8k) :: x(nsample, ndim), x2(nsample, ndim)
    real(r8k) :: y(nsample), dj(ndim), b(ndim), djinv(ndim)
    real(r8k) :: wk(ncomp), sk(ncomp)
    real(r8k) :: s, s2
    integer(i4k) :: i, nprn
    real(r8k) :: obj, bgrad(ndim), wgrad(ncomp), s2grad!, agrad(ncomp)
    integer(i8k) :: starttime, endtime, timerate

    s = 0.9d0
    s2 = s ** d_two
!    sk = get_mixgauss_scale(ncomp, d_two)
    sk = [0.000d0, 0.035d0, 0.072d0, 0.110d0, 0.149d0,  &
          0.189d0, 0.231d0, 0.275d0, 0.320d0, 0.366d0,  &
          0.414d0, 0.464d0, 0.516d0, 0.569d0, 0.625d0,  &
          0.682d0, 0.741d0, 0.803d0, 0.866d0, 0.932d0]
!    wk = get_mixgauss_weight(ncomp, 0.7 * d_one)
    wk = [0.715d0, 0.015d0, 0.015d0, 0.015d0, 0.015d0,  &
          0.015d0, 0.015d0, 0.015d0, 0.015d0, 0.015d0,  &
          0.015d0, 0.015d0, 0.015d0, 0.015d0, 0.015d0,  &
          0.015d0, 0.015d0, 0.015d0, 0.015d0, 0.015d0]
!    call get_predictors(nsample, ndim, x)
!    call get_sparse_coef(ndim, ncausal, b)
!    call get_responses(nsample, ndim, x, b, s2, y)

    ! output data into a file 
!    open(101, file = 'responses.dat', status='new')
!    write(101, *) y
!    close(101)
!    open(102, file = 'coefs.dat', status='new')
!    write(102, *) b
!    close(102)
!    open(103, file = 'predictors.dat', status='new')
!    write(103, *) x
!    close(103)
!    read data from file
    open(201, file = 'responses.dat', status = 'old')
    read(201, *) y
    close(201)
    open(202, file = 'coefs.dat', status='old')
    read(202, *) b
    close(202)
    open(203, file = 'predictors.dat', status='old')
    read(203, *) x
    close(203)

!    call deepcopy_matrix(X, Xcopy1, nsample, ndim)
!    call deepcopy_matrix(X, Xcopy2, nsample, ndim)

    x2 = x ** d_two
    dj = sum(x2, 1)
    do i = 1, ndim
        djinv(i) = d_one / dj(i)
    end do

    obj = d_zero
    bgrad = d_zero
    wgrad = d_zero
    s2grad = d_zero
    write (6, *) "Start objective calculation."
    call system_clock(starttime, timerate)
    do i = 1, 1000
        call plr_obj_grad_shrinkop(nsample, ndim, x, y, b, s2, ncomp, wk, sk, djinv, obj, bgrad, wgrad, s2grad)
    end do
    write (6, *) "Done."
    call system_clock(endtime)
    write (6, *) "Time required for 1000 function calls: ", real(endtime - starttime) / real(timerate)

    nprn = 5
    write (6, *) "Standard deviation of ASH components"
    call print_vector(sk, ncomp)

    write (6, *) "Initial scale components"
    call print_vector(wk, ncomp)

    write (6, *) "Top left corner of X"
    call print_array2d(x(1:nprn, 1:nprn), nprn, nprn)

    write (6, *) "dj"
    call print_vector(dj(1:nprn), nprn)

    write (6, *) "djinv"
    call print_vector(djinv(1:nprn), nprn)

    write (6, *) "True coefficients"
    call print_vector(b(1:nprn), nprn)

    write (6, *) "Responses"
    call print_vector(y(1:nprn), nprn)

end program
