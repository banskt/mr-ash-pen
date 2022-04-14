use, intrinsic :: iso_fortran_env, only : real32, real64
real(kind = 8) :: a1, a2, a3

print 1, "real32", RADIX(1._real32), EXPONENT(TINY(1._real32)), EXPONENT(HUGE(1._real32))
print 1, "real64", RADIX(1._real64), EXPONENT(TINY(1._real64)), EXPONENT(HUGE(1._real64))

1 FORMAT (A," has radix ", I0, " with exponent range ", I0, " to ", I0, ".")

a1 = -1022.d0
print *, a1
a2 = dexp(a1)
print *, a2
a3 = log(a2)
print *, a3

end
