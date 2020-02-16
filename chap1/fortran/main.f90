
program heat1d
    implicit none
    integer :: i, n, total_timestep
    integer, parameter :: Nmax = 100    ! fortran does not distinguish "n" and "N"
    double precision, parameter :: k=1.0d0, t_end =1.0d0, boundary_x0 = 1.0d0, boundary_xN=0.0d0
    double precision :: dx, dt, alpha
    double precision, allocatable :: phi(:, :)
    character(len=64) :: cFileName

    dx =1.0d0 / dble(Nmax)
    dt = 0.1d0 * dx**2
    alpha =k * dt / (dx**2)

    total_timestep = int(t_end / dt)

    allocate(phi(Nmax+2, total_timestep))

    do i = 2, Nmax + 1
        phi(i, 1) = dble(i-1) / dble(Nmax)
    end do

    do n = 1, total_timestep - 1
        phi(1, n) = boundary_x0
        phi(Nmax+2, n) = boundary_xN
        do i=2, Nmax+1
            phi(i, n+1) = phi(i,n) + alpha * (phi(i+1,n) - 2.0d0*phi(i,n) + phi(i-1,n))
        end do
    end do

    cFileName = "output.txt"
    open(unit=1, file=cFileName, status='unknown')
        do n=1, total_timestep
            if (mod(n, 100) == 0) then
                do i=2, Nmax+1
                    write(1,"(2(2x,f22.14))") dble(i-1)*dx, phi(i,n)
                end do
                write(1,"('')")
                write(1,"('')")
            end if
        end do
    close(1)

end program

