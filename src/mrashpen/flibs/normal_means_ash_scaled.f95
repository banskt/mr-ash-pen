module nmash_scaled
    use env_precision
    implicit none
    private
    public logML

contains

    subroutine calculate_logML()
    end subroutine

    subroutine calculate_logLjk()
        real(r8k) :: 
    end subroutine calculate_logLjk

end module nmash_scaled

subroutine a_plus_matB(a, b, m, n, res)
    integer(i4k), intent(in) :: m, n
    real(r8k), intent(in)    :: a
    real(r8k), intent(in)    :: b(m, n)
    real(r8k), intent(out)   :: res(m, n)
    integer(i4k)             :: i, j
    do j = 1, n
        do i = 1, m
            res(i, j) = b(i, j) + a
        end do
    end do
end subroutine a_plus_matB

