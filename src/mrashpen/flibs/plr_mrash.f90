module plr_mrash
    use env_precision
    use global_parameters
    use futils, only: duplicate_columns
    use normal_means_ash_scaled!, only: normal_means_ash_lml
    implicit none
!    private
!    public plr_obj_grad_shrinkop
!
contains
!
   subroutine plr_obj_grad_shrinkop(n, p, X, y, b, s2, k, wk, sk, djinv,            &
                                   obj, bgrad, wgrad, s2grad)
        implicit none
        integer(i4k), intent(in)   :: n, p, k
        real(r8k), intent(in)      :: X(n, p), y(n), b(p)
        real(r8k), intent(in)      :: s2
        real(r8k), intent(in)      :: wk(k), sk(k)
        real(r8k), intent(in)      :: djinv(p)
        real(r8k), intent(out)     :: obj
        real(r8k), intent(out)     :: bgrad(p), wgrad(k), s2grad
!       Functions
        real(r8k), external        :: ddot
!
!       NMash variables
        real(r8k), dimension(p)    :: lml
        real(r8k), dimension(p)    :: lml_bd,  lml_bd_bd
        real(r8k), dimension(p, k) :: lml_wd,  lml_bd_wd
        real(r8k), dimension(p)    :: lml_s2d, lml_bd_s2d
!       local variables (shrinkage operator)
        real(r8k), dimension(p)    :: Mb, Mb_bgrad, Mb_s2grad
        real(r8k), dimension(p, k) :: Mb_wgrad
!       local variables (penalty operator)
        real(r8k), dimension(p)    :: lambdaj
        real(r8k), dimension(p)    :: l_bgrad, l_s2grad
        real(r8k), dimension(k)    :: l_wgrad
!       local variable (others)
        real(r8k), dimension(n)    :: r, XMb
        real(r8k)                  :: rTr
        real(r8k), dimension(p)    :: bvar, rTX
        real(r8k), dimension(k)    :: v1 !, v2
!
!        integer :: i, j
!
        bvar = s2 * djinv
!
!       ========================
!       Use normal means model
!       ========================
!       Initialize
        lml = d_zero
        lml_bd = d_zero
        lml_wd = d_zero
        lml_s2d = d_zero
        lml_bd_bd = d_zero
        lml_bd_wd = d_zero
        lml_bd_s2d = d_zero
        call normal_means_ash_lml(p, k, b, s2, wk, sk, djinv,                        &
                                  lml, lml_bd, lml_wd, lml_s2d,                      &
                                  lml_bd_bd, lml_bd_wd, lml_bd_s2d)
!
!       ========================
!       Objective function
!       ========================
        call plr_shrinkage_operator(b, bvar, djinv,                                  &
                                    lml_bd, lml_bd_bd, lml_bd_wd, lml_bd_s2d,        &
                                    Mb, Mb_bgrad, Mb_wgrad, Mb_s2grad)
        call plr_penalty_operator(b, bvar, djinv,                                    &
                                  lml, lml_bd, lml_wd, lml_s2d,                      &
                                  lml_bd_bd, lml_bd_wd, lml_bd_s2d,                  &
                                  lambdaj, l_bgrad, l_wgrad, l_s2grad)
        XMb = d_zero
        call dgemv('N', n, p, d_one, X, n, Mb, 1, d_zero, XMb, 1)
        r    = y - XMb ! vector addition, F90
        rTr  = ddot(n, r, 1, r, 1)
        obj  = (d_half * rTr / s2)                                                   &
               + sum(lambdaj)                                                        &
               + d_half * (n - p) * (log2pi + log(s2))
!
!       ========================
!       Gradients
!       ========================
!       Gradient with respect to b
        rTX = d_zero
        call dgemv('T', n, p, d_one, X, n, r, 1, d_zero, rTX, 1)
        bgrad  = - (rTX * Mb_bgrad / s2) + l_bgrad
!
!       Gradient with respect to w
        v1 = d_zero
!        write (6, *) 'v1 =>'
!        call print_vector(v1, k)
        call dgemv('T', p, k, d_one, Mb_wgrad, p, rTX, 1, d_zero, v1, 1)
!        call print_vector(v1, k)
        wgrad  = - (v1 / s2) + l_wgrad
!        write (6, *) "subroutine wgrad"
!        call print_vector(wgrad, k)
!
!       Gradient with respect to s2
        s2grad = - d_half * (rTr / (s2 * s2))                                        &
                 - ddot(p, rTX, 1, Mb_s2grad, 1) / s2                                &
                 + sum(l_s2grad)                                                     &
                 + d_half * (n - p) / s2
!        write (6, *) "subroutine s2grad"
!        write (6, *) s2grad

    end subroutine plr_obj_grad_shrinkop
!
!
    subroutine plr_shrinkage_operator(b, bvar, djinv,                                &
                                      lml_bd, lml_bd_bd, lml_bd_wd, lml_bd_s2d,      &
                                      Mb, Mb_bgrad, Mb_wgrad, Mb_s2grad)
        use env_precision
        use global_parameters
        implicit none
        real(r8k), intent(in)  :: b(:), bvar(:)
        real(r8k), intent(in)  :: djinv(:)
        real(r8k), intent(in)  :: lml_bd(:), lml_bd_bd(:)
        real(r8k), intent(in)  :: lml_bd_wd(:, :)
        real(r8k), intent(in)  :: lml_bd_s2d(:)
        real(r8k), intent(out) :: Mb(size(b)), Mb_bgrad(size(b))
        real(r8k), intent(out) :: Mb_wgrad(size(b), size(lml_bd_wd, 2))
        real(r8k), intent(out) :: Mb_s2grad(size(b))
!       local variables
        integer(i4k) :: p, k
        real(r8k), allocatable :: bvar_mat(:, :)
        !real(r8k), allocatable :: v_one(:), bvar_mat(:, :)
!
!       internal placeholders
        p     = size(b)
        k     = size(lml_bd_wd, 2)
        !if( allocated(v_one) )  deallocate( v_one )
        if( allocated(bvar_mat) )  deallocate( bvar_mat )
        !allocate (v_one(p), bvar_mat(p, k))
        allocate(bvar_mat(p, k))
        !v_one = d_one
        call duplicate_columns(bvar, k, bvar_mat)
!    
!       calculation
        Mb        = b + (bvar * lml_bd)
        Mb_bgrad  = d_one + (bvar * lml_bd_bd)
        Mb_wgrad  = bvar_mat * lml_bd_wd
        Mb_s2grad = (lml_bd * djinv) + (bvar * lml_bd_s2d)
    end subroutine
!
!
    subroutine plr_penalty_operator(b, bvar, djinv,                                  &
                                    lml, lml_bd, lml_wd, lml_s2d,                    &
                                    lml_bd_bd, lml_bd_wd, lml_bd_s2d,                &
                                    lambdaj, l_bgrad, l_wgrad, l_s2grad)
        use env_precision
        use global_parameters
        implicit none
        real(r8k), intent(in)  :: b(:), bvar(:)
        real(r8k), intent(in)  :: djinv(:)
        real(r8k), intent(in)  :: lml(:), lml_bd(:), lml_bd_bd(:)
        real(r8k), intent(in)  :: lml_wd(:, :), lml_bd_wd(:, :)
        real(r8k), intent(in)  :: lml_s2d(:), lml_bd_s2d(:)
        real(r8k), intent(out) :: lambdaj(size(b)), l_bgrad(size(b))
        real(r8k), intent(out) :: l_wgrad(size(lml_wd, 2))
        real(r8k), intent(out) :: l_s2grad(size(b))
!       local variables
        integer(i4k) :: p, k
        real(r8k), allocatable :: M1(:, :), M2(:, :), M3(:, :)
        real(r8k), allocatable :: v1(:)
!
!       internal placeholders
        p     = size(b)
        k     = size(lml_wd, 2)
        if( allocated(M1) )  deallocate(M1)
        if( allocated(M2) )  deallocate(M2)
        if( allocated(M3) )  deallocate(M3)
        if( allocated(v1) )  deallocate(v1)
        allocate(M1(p, k), M2(p, k), M3(p, k))
        allocate(v1(p))
!
!       Calculation
        lambdaj  = - lml - d_half * bvar * lml_bd * lml_bd
!       Gradient with respect to b
        l_bgrad  = - lml_bd - bvar * lml_bd * lml_bd_bd
!       Gradient with respect to w
        call duplicate_columns(bvar, k, M1)
        call duplicate_columns(lml_bd, k, M2)
        M3       = - lml_wd - (M1 * M2 * lml_bd_wd)
        l_wgrad  = sum(M3, 1)
!       Gradient with respect to s2
        v1       = bvar * lml_bd * lml_bd_s2d
        l_s2grad = - (lml_s2d + v1 + d_half * lml_bd * lml_bd * djinv)
    end subroutine plr_penalty_operator
end module 
