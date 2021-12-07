program main
    use nmash_scaled
    use env_precision
    use global_parameters
    implicit none
    integer, parameter :: ndim = 5, ncomp = 3
    real(r8k) :: y(ndim), dj(ndim), logML(ndim)
    real(r8k) :: wk(ncomp), sk(ncomp)
    real(r8k) :: s
    integer(i4k) :: i

    s = 0.9d0
    y = [3.5d0, 4.5d0, 1.2d0, 6.5d0, 2.8d0]
    do i = 1, ndim
        dj(i) = d_one
        logML(i) = d_zero
    end do
    sk = [0.d0, 0.1d0, 0.9d0]
    wk = [0.5d0, 0.25d0, 0.25d0]
    write (6, *) "Input data =>"
    call print_vector(y, ndim)
    call initialize(y, s, wk, sk, dj, ndim, ncomp)
    logML = calculate_logML()
    write (6, *) "logML =>"
    call print_vector(logML, ndim)
end program 
