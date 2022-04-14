module optutils
    use env_precision
    use global_parameters
    implicit none
!
contains

    subroutine combine_parameters(x, t, w, s, is_topt, is_wopt, is_sopt)
!
!   x = [t_1, t_2, ... t_p, w_1, w_2, ..., w_k, s]
!   This subroutine helps to combine array t, array w, and variable s
!   to an array of DOUBLE PRECISION variables x.
!       x contains t only if is_topt = .true.
!       x contains w only if is_wopt = .true.
!       x contains s only if is_sopt = .true.
!   nparams is an INTEGER which contains the total number of variables in x.
!   p is an INTEGER; size of t
!   k is an INTEGER; size of w
!
!   see combine_integer_parameters() for combining INTEGER variables.
!
        implicit none
        integer(i4k) :: p, k, iopt
        real(r8k)    :: x(:), t(:), w(:), s
        logical      :: is_topt, is_wopt, is_sopt
!
        p = size(t)
        k = size(w)
        iopt = 0
        if (is_topt) then
            x(1:p) = t
            iopt = p
        end if
        if (is_wopt) then
            x(iopt+1:iopt+k) = w
            iopt = iopt + k
        end if
        if (is_sopt) then
            x(iopt+1) = s
        end if
    end subroutine
!
    subroutine combine_integer_parameters(x, t, w, s, is_topt, is_wopt, is_sopt)
!
!   same as combine_parameters() but for INTEGER variables in t, w, s and x.
!
!   x = [t_1, t_2, ... t_p, w_1, w_2, ..., w_k, s]
!   This subroutine helps to combine array t, array a, and variable s
!   to an array of INTEGER variables x.
!       x contains t only if is_topt = .true.
!       x contains w only if is_wopt = .true.
!       x contains s only if is_sopt = .true.
!   nparams is an INTEGER which contains the total number of variables in x.
!   p is an INTEGER; size of t
!   k is an INTEGER; size of w
!
        implicit none
        integer(i4k) :: p, k, iopt
        integer(i4k) :: x(:), t(:), w(:), s
        logical      :: is_topt, is_wopt, is_sopt
!
        p = size(t)
        k = size(w)
        iopt = 0
        if (is_topt) then
            x(1:p) = t
            iopt = p
        end if
        if (is_wopt) then
            x(iopt+1:iopt+k) = w
            iopt = iopt + k
        end if
        if (is_sopt) then
            x(iopt+1) = s
        end if
    end subroutine
!
    subroutine split_parameters(x, t, w, s, is_topt, is_wopt, is_sopt)
!
!   x = [t_1, t_2, ... t_p, w_1, w_2, ..., w_k, s]
!   This subroutine helps to split an array of DOUBLE PRECISION variables x
!   to its constituents: array t, array w, and variable s.
!       x contains t only if is_topt = .true.
!       x contains w only if is_wopt = .true.
!       x contains s only if is_sopt = .true.
!   nparams is an INTEGER which contains the total number of variables in x.
!   p is an INTEGER; size of t
!   k is an INTEGER; size of w
!
        implicit none
        integer(i4k) :: p, k, iopt
        real(r8k)    :: x(:), t(:), w(:), s
        logical      :: is_topt, is_wopt, is_sopt
!
        p = size(t)
        k = size(w)
        iopt = 0
        if (is_topt) then
            t = x(1:p)
            iopt = p
        end if
        if (is_wopt) then
            w = x(iopt+1:iopt+k)
            iopt = iopt + k
        end if
        if (is_sopt) then
            s = x(iopt+1)
        end if
    end subroutine
!
end module
