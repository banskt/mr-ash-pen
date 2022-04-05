program main
    use env_precision
    use global_parameters
    use futils
    use get_data
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

    s = 0.9d0
    s2 = s ** d_two
    sk = get_mixgauss_scale(ncomp, d_two)
    wk = get_mixgauss_weight(ncomp, 0.7 * d_one)
    x  = get_predictors(nsample, ndim)
    b  = get_sparse_coef(ndim, ncausal)
    y  = get_responses(nsample, ndim, x, b, s2)

!    call deepcopy_matrix(X, Xcopy1, nsample, ndim)
!    call deepcopy_matrix(X, Xcopy2, nsample, ndim)

    x2 = x ** d_two
    dj = sum(x2, 1)
    do i = 1, ndim
        djinv(i) = d_one / dj(i)
    end do

    obj = d_zero
    call fill_real_vector(bgrad, d_zero)
    call fill_real_vector(wgrad, d_zero)
    s2grad = d_zero
    write (6, *) "Start objective calculation."
    do i = 1, 1000
        call plr_obj_grad_shrinkop(nsample, ndim, x, y, b, s2, ncomp, wk, sk, djinv, obj, bgrad, wgrad, s2grad)
    end do
    write (6, *) "Done."

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
