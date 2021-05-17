subroutine tls_phonon(progress, int_progress)
    implicit none
    external progress
    integer, intent(in) :: int_progress
    integer :: i = 0, j, progress

    do i =1,1000
        if (modulo(i,int_progress) .eq. 0) then
            j = progress(int_progress)
        end if
    end do

end subroutine tls_phonon